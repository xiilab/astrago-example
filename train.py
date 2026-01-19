import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.models import resnet50, ResNet50_Weights
import argparse

# ✅ LazyFakeDataset: 메모리 아끼는 가짜 데이터셋
class LazyFakeDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(3, 224, 224)            # 각 샘플마다 랜덤 이미지 생성
        y = torch.randint(0, 100, (1,)).item()  # 0~99 사이의 정수 label
        return x, y


def setup(rank, world_size, use_cuda):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    # CPU는 gloo 백엔드, GPU는 nccl 백엔드 사용
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if use_cuda:
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args, use_cuda):
    setup(rank, world_size, use_cuda)

    # 디바이스 설정
    if use_cuda:
        device = torch.device(f"cuda:{rank}")
        print(f"[Rank {rank}] Starting training on GPU {rank}")
    else:
        device = torch.device("cpu")
        print(f"[Rank {rank}] Starting training on CPU")

    start_time = time.time()

    # 모델 정의 및 분산 래핑
    # weights=None: 폐쇄망 환경에서도 작동하도록 pretrained weights 다운로드 비활성화
    model = resnet50(weights=None, num_classes=args.num_classes).to(device)

    if world_size > 1:
        if use_cuda:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        else:
            model = nn.parallel.DistributedDataParallel(model)

    # ✅ Lazy 데이터셋: args.dataset_size 샘플
    dataset = LazyFakeDataset(args.dataset_size)
    pin_memory = use_cuda  # GPU일 때만 pin_memory 사용
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=pin_memory)
    else:
        sampler = None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        model.train()

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"[Rank {rank}] Epoch {epoch:3d} - Avg Loss: {avg_loss:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[Rank {rank}] Training completed in {total_time:.2f} seconds")

    # 모델 저장 (rank 0에서만)
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "model_final.pth")
        # DDP로 래핑된 경우 model.module, 아닌 경우 model 자체 저장
        state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
        torch.save(state_dict, save_path)
        print(f"[Rank 0] Model saved to {save_path}")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DDP ResNet50 AstraGo 워크로드 테스트용")
    parser.add_argument('--epochs', type=int, default=100, help='학습 에폭 수 (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64, help='배치 크기 (default: 64)')
    parser.add_argument('--dataset-size', type=int, default=100000, help='데이터셋 샘플 수 (default: 100000)')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률 (default: 0.001)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader num_workers (default: 4)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='모델 저장 디렉토리 (default: ./checkpoints)')
    parser.add_argument('--num-classes', type=int, default=100, help='클래스 개수 (default: 100)')
    parser.add_argument('--cpu', action='store_true', help='CPU 모드로 강제 실행')
    args = parser.parse_args()

    gpu_count = torch.cuda.device_count()
    print(f"Total GPUs available: {gpu_count}")

    # CPU 모드 결정: --cpu 옵션 또는 GPU가 없는 경우
    use_cuda = not args.cpu and gpu_count > 0

    if use_cuda:
        world_size = gpu_count
        if world_size == 1:
            print("Single GPU detected. Running in single-GPU mode.")
        else:
            print(f"Multiple GPUs detected ({world_size}). Running in DDP mode.")
    else:
        world_size = 1
        if args.cpu:
            print("CPU mode enabled by --cpu flag.")
        else:
            print("No GPU available. Running in CPU mode.")

    mp.spawn(train, args=(world_size, args, use_cuda), nprocs=world_size, join=True)