"""MNIST 분산학습 예제 (PyTorch DDP + torchrun).

AstraGo v2의 분산학습(Kubeflow Trainer v2 TrainJob) 워크로드 검증용 예제.

핵심 설계
---------
- torchrun 컨트랙트로 작성: 프로세스 스폰과 분산 좌표(RANK/LOCAL_RANK/
  WORLD_SIZE/MASTER_ADDR/MASTER_PORT) 주입은 torchrun이 담당하고, 이 스크립트는
  환경변수만 읽어 ``init_process_group`` 을 호출한다. ``mp.spawn`` 을 쓰지 않으므로
  단일 노드 멀티 GPU는 물론 멀티 노드까지 코드 변경 없이 확장된다.
- 오프라인: 데이터셋은 레포에 번들한 ``.gz`` (idx-ubyte)를 직접 읽고, 모델은
  코드로 정의한 소형 CNN이므로 인터넷/사전학습 가중치 다운로드가 전혀 없다.
- 환경변수가 없으면(``python mnist_ddp.py``) 단일 프로세스로 fallback 한다.

실행 예시
---------
    # 단일 프로세스 (torchrun 없이)
    python mnist_ddp.py --epochs 5

    # 단일 노드 N GPU (또는 CPU 프로세스 N개)
    torchrun --nproc_per_node=2 --standalone mnist_ddp.py --epochs 5

    # AstraGo TrainJob: 백엔드가 위 torchrun 커맨드를 자동 생성한다.
"""

import argparse
import gzip
import os
import struct
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# MNIST 정규화 상수 (평균/표준편차)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049


# --------------------------------------------------------------------------- #
# 데이터: torchvision 없이 idx-ubyte(.gz)를 직접 읽는 오프라인 로더
# --------------------------------------------------------------------------- #
def _read_idx_images(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, num, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != IMAGE_MAGIC:
        raise ValueError(f"이미지 매직 넘버 불일치({magic} != {IMAGE_MAGIC}): {path}")
    # bytearray 로 감싸 writable 버퍼로 만들어 frombuffer 경고를 피한다.
    tensor = torch.frombuffer(bytearray(data[16:]), dtype=torch.uint8)
    return tensor.view(num, rows, cols)


def _read_idx_labels(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, num = struct.unpack(">II", data[:8])
    if magic != LABEL_MAGIC:
        raise ValueError(f"라벨 매직 넘버 불일치({magic} != {LABEL_MAGIC}): {path}")
    return torch.frombuffer(bytearray(data[8:]), dtype=torch.uint8).view(num).long()


def _resolve_raw_dir(data_dir: str) -> str:
    """``<data_dir>/MNIST/raw`` 우선, 없으면 ``<data_dir>`` 를 원본 디렉토리로 본다."""
    nested = os.path.join(data_dir, "MNIST", "raw")
    if os.path.isdir(nested):
        return nested
    return data_dir


class MnistGzDataset(Dataset):
    """번들된 MNIST ``.gz`` 를 읽어 정규화된 (1,28,28) 텐서를 반환한다."""

    def __init__(self, data_dir: str, train: bool = True):
        raw_dir = _resolve_raw_dir(data_dir)
        prefix = "train" if train else "t10k"
        images_path = os.path.join(raw_dir, f"{prefix}-images-idx3-ubyte.gz")
        labels_path = os.path.join(raw_dir, f"{prefix}-labels-idx1-ubyte.gz")
        for path in (images_path, labels_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"MNIST 데이터가 없습니다: {path}\n"
                    f"레포에 번들된 data/MNIST/raw/*.gz 를 확인하세요."
                )
        self.images = _read_idx_images(images_path)
        self.labels = _read_idx_labels(labels_path)
        if len(self.images) != len(self.labels):
            raise ValueError("이미지 수와 라벨 수가 일치하지 않습니다.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index].float().div_(255.0).sub_(MNIST_MEAN).div_(MNIST_STD)
        return image.unsqueeze(0), int(self.labels[index])


# --------------------------------------------------------------------------- #
# 모델: 표준 MNIST CNN (다운로드 불필요)
# --------------------------------------------------------------------------- #
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 28 -> 26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 26 -> 24
        self.pool = nn.MaxPool2d(2)                     # 24 -> 12
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


# --------------------------------------------------------------------------- #
# 분산 초기화 / 정리
# --------------------------------------------------------------------------- #
def is_distributed() -> bool:
    return "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(use_cuda: bool):
    """torchrun 이 주입한 env(RANK/WORLD_SIZE/MASTER_ADDR/PORT)로 프로세스 그룹 초기화."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    if use_cuda:
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST 분산학습 (PyTorch DDP + torchrun)")
    parser.add_argument("--epochs", type=int, default=5, help="학습 에폭 수 (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="프로세스당 배치 크기 (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률 (default: 1e-3)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="MNIST 데이터 디렉토리 (default: ./data)")
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                        help="모델 저장 디렉토리. OUTPUT_DIR env 가 있으면 우선 (default: ./checkpoints)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader num_workers (default: 2)")
    parser.add_argument("--cpu", action="store_true", help="CPU 모드로 강제 실행")
    args = parser.parse_args()

    # 스크립트 위치 기준으로 상대경로 해석 (실행 cwd 와 무관하게 데이터를 찾도록)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.normpath(os.path.join(script_dir, args.data_dir))
    return args


def main():
    args = parse_args()

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    distributed = is_distributed()

    if distributed:
        rank, world_size, local_rank = setup_distributed(use_cuda)
    else:
        rank, world_size, local_rank = 0, 1, 0

    device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")
    is_main = rank == 0

    def log(message: str):
        if is_main:
            print(message, flush=True)

    log("=" * 60)
    log("  MNIST 분산학습 (PyTorch DDP)")
    log("=" * 60)
    log(f"  분산 모드   : {'ON' if distributed else 'OFF (단일 프로세스)'}")
    log(f"  world_size  : {world_size}")
    log(f"  디바이스    : {'GPU (cuda)' if use_cuda else 'CPU'}")
    log(f"  데이터      : {args.data_dir}")
    log(f"  에폭        : {args.epochs}")
    log(f"  배치(프로세스당) : {args.batch_size}")
    log("=" * 60)

    dataset = MnistGzDataset(args.data_dir, train=True)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    model = Net().to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        num_batches = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        if distributed:
            # 백엔드 무관하게 동작하도록 SUM 후 world_size 로 나눈다 (gloo AVG 미지원 대비).
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor / world_size).item()
        log(f"[Epoch {epoch:3d}/{args.epochs}] loss = {avg_loss:.4f}")

    log(f"학습 완료: {time.time() - start:.1f}s")

    # 모델 저장은 rank 0 에서만. OUTPUT_DIR(AstraGo 주입) 우선.
    if is_main:
        save_dir = os.environ.get("OUTPUT_DIR") or args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        state_dict = model.module.state_dict() if distributed else model.state_dict()
        save_path = os.path.join(save_dir, "mnist_cnn.pt")
        torch.save(state_dict, save_path)
        log(f"모델 저장 완료: {save_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
