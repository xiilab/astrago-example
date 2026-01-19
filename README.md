# ResNet DDP 분산 학습 예제 (AstraGo 워크로드 테스트용)

이 코드는 AstraGo의 워크로드 생성 및 모니터링 기능을 테스트하기 위한 PyTorch 기반의 분산 학습(Distributed Data Parallel, DDP) 예제입니다.  
실제 데이터가 아닌, 메모리를 아끼는 가짜 데이터셋(LazyFakeDataset)을 사용하여 여러 GPU에서 ResNet-50 모델을 학습합니다.

## 주요 특징

- **CPU/GPU 모두 지원**: CPU, 단일 GPU, 멀티 GPU 환경 모두 지원
- **폐쇄망 환경 지원**: pretrained weights 다운로드 없이 동작 (인터넷 연결 불필요)
- **PyTorch DDP(DistributedDataParallel) 기반 멀티 GPU 학습**
- **SyncBatchNorm** 적용 (멀티 GPU 환경에서만)
- **가짜 데이터셋(LazyFakeDataset)**: 실제 이미지 대신 무작위 텐서와 라벨을 생성하여 메모리 사용 최소화
- **100,000개 샘플, 100 에폭 학습 (인자로 조정 가능)**
- **각 프로세스별 학습 진행 상황 및 평균 Loss 출력**
- **학습 종료 후 모델 파일(model_final.pth) 저장**
- **AstraGo 환경에서 워크로드 생성/모니터링 테스트에 적합**

## 파일 구조

```
train.py  # 분산 학습 전체 로직이 포함된 메인 파일
```

## 실행 환경

- Python 3.8 이상
- PyTorch 1.9 이상
- torchvision
- CPU 또는 CUDA가 지원되는 NVIDIA GPU
- **폐쇄망 환경에서도 동작** (인터넷 연결 불필요)

## 설치 방법

```bash
pip install torch torchvision
```

## 실행 방법

1. **1개 이상의 GPU가 필요합니다.** (단일 GPU 또는 멀티 GPU 모두 지원)
2. 아래 명령어로 실행하세요.

```bash
python train.py [옵션들]
```

### 주요 명령줄 인자

| 인자명           | 설명                              | 기본값           |
|------------------|-----------------------------------|------------------|
| --epochs         | 학습 에폭 수                      | 100              |
| --batch-size     | 배치 크기                         | 64               |
| --dataset-size   | 데이터셋 샘플 수                  | 100000           |
| --lr             | 학습률                            | 0.001            |
| --num-workers    | DataLoader num_workers            | 4                |
| --save-dir       | 모델 저장 디렉토리                | ./checkpoints    |
| --num-classes    | 분류 클래스 개수                  | 100              |
| --cpu            | CPU 모드로 강제 실행              | False            |

### 실행 예시

```bash
# GPU 사용 (자동 감지)
python train.py --epochs 50 --batch-size 128 --dataset-size 50000 --lr 0.01 --save-dir ./output

# CPU 모드로 강제 실행
python train.py --cpu --epochs 10 --batch-size 32 --dataset-size 10000 --save-dir ./output
```

- GPU가 없을 경우, 자동으로 CPU 모드로 학습이 진행됩니다.
- GPU가 1개일 경우, 단일 GPU 모드로 학습이 진행됩니다.
- GPU가 2개 이상이면, DDP 모드로 각 GPU에서 프로세스가 생성되어 분산 학습이 시작됩니다.
- `--cpu` 옵션을 사용하면 GPU가 있어도 CPU 모드로 강제 실행됩니다.

### 모델 저장

- 학습이 끝나면 **rank 0 프로세스**에서만 지정한 디렉토리(`--save-dir`)에 `model_final.pth`로 모델이 저장됩니다.

## 코드 설명

- **LazyFakeDataset**: `__getitem__`에서 무작위 이미지(3x224x224)와 0~99 사이의 라벨을 생성합니다.
- **setup/cleanup**: DDP 환경 초기화 및 정리 함수입니다.
- **train**: 각 GPU별로 모델, 데이터로더, 옵티마이저, 손실함수 등을 생성하고 지정한 에폭 동안 학습을 수행합니다.
- **main**: 사용 가능한 GPU 개수를 확인하고, 2개 이상일 때만 `mp.spawn`으로 분산 학습을 시작합니다.

## 참고

- 이 코드는 실제 데이터가 아닌 가짜 데이터를 사용하므로, 모델의 성능이나 정확도는 의미가 없습니다.
- AstraGo의 워크로드 생성 및 모니터링 기능을 테스트하는 용도로 사용하세요.
