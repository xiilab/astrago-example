# KoAlpaca 파인튜닝 + ResNet DDP 학습 예제

AstraGo 워크로드 테스트를 위한 PyTorch 기반 학습 예제 모음입니다.

---

## 1. KoAlpaca 파인튜닝 (LLM)

`HuggingFaceTB/SmolLM2-135M` 모델을 KoAlpaca 데이터셋으로 파인튜닝하는 예제입니다.  
LoRA(Low-Rank Adaptation)를 사용하여 적은 GPU 메모리로도 학습할 수 있습니다.

### 주요 특징

- **SmolLM2-135M**: 경량 LLM (135M 파라미터, Git LFS 불필요)
- **beomi/KoAlpaca-v1.1a**: 한국어 Instruction-Following 데이터셋 (21,155 샘플)
- **LoRA 기반 효율적 파인튜닝** (PEFT)
- **폐쇄망 지원**: 모델, 데이터셋, 패키지 모두 레포에 포함 (pip install 불필요)
- **GPU 자동 최적화**: FP16/BF16 자동 감지, torch.compile, pin_memory 등
- **CPU / 단일 GPU / 멀티 GPU 지원**

### 파일 구조

```
├── model/               # SmolLM2-135M (safetensors 분할, 파일당 <100MB)
├── dataset/             # KoAlpaca-v1.1a
├── packages/            # Python 패키지 (torch 제외, pip install 불필요)
├── finetune.py          # 파인튜닝 메인 스크립트
├── train.py             # ResNet DDP 학습 (기존)
└── .gitignore
```

### 사용법

#### 빠른 실행

모델, 데이터셋, Python 패키지가 모두 레포에 포함되어 있으므로 폐쇄망에서도 바로 실행 가능합니다.

```bash
# 바로 실행 (pip install 불필요, packages/ 자동 참조)
# 기본: 5,000 샘플, 1 에폭, GPU 자동 FP16/BF16 (~313 스텝)
python finetune.py
```

#### 전체 데이터 학습

```bash
# 전체 21,155 샘플, 3 에폭 (~3,969 스텝)
python finetune.py --max-samples -1 --epochs 3
```

#### 커스텀 설정

```bash
python finetune.py \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 3e-4 \
  --max-length 512 \
  --max-samples 10000
```

#### 기타

```bash
# Full Fine-tuning (LoRA 비활성화)
python finetune.py --no-lora

# CPU 모드
python finetune.py --cpu --max-samples 100 --epochs 1

# HuggingFace에서 직접 다운로드하여 사용 (온라인 환경)
python finetune.py \
  --model-path HuggingFaceTB/SmolLM2-135M \
  --dataset-path beomi/KoAlpaca-v1.1a
```

#### 추론 (vLLM)

파인튜닝된 모델은 vLLM으로 서빙합니다.

```bash
vllm serve ./finetuned-model/final
```

### 파인튜닝 주요 옵션

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--model-path` | 모델 경로 또는 HF 모델명 | `./model` |
| `--dataset-path` | 데이터셋 경로 또는 HF 데이터셋명 | `./dataset` |
| `--output-dir` | 결과 저장 경로 | `./finetuned-model` |
| `--epochs` | 학습 에폭 수 | `1` |
| `--batch-size` | 배치 크기 | `8` |
| `--gradient-accumulation-steps` | 그래디언트 누적 | `2` |
| `--learning-rate` | 학습률 | `2e-4` |
| `--max-length` | 최대 토큰 길이 | `256` |
| `--max-samples` | 학습 데이터 최대 샘플 수 (`-1`=전체) | `5000` |
| `--use-lora` / `--no-lora` | LoRA 사용 여부 | LoRA ON |
| `--lora-r` | LoRA rank | `8` |
| `--lora-alpha` | LoRA alpha | `16` |
| `--fp16` / `--bf16` | 혼합 정밀도 (GPU시 자동 활성화) | 자동 |
| `--cpu` | CPU 강제 실행 | OFF |

### 품질 강화 가이드

기본 설정은 빠른 테스트용으로 최적화되어 있습니다. 모델 출력 품질을 높이려면 아래 방법을 조합하세요.

#### 레벨 1: 기본 강화 (권장 시작점)

```bash
python finetune.py \
  --max-samples -1 \
  --epochs 3 \
  --max-length 512 \
  --learning-rate 1e-4
```

| 변경 | 기본값 | 강화값 | 효과 |
|------|--------|--------|------|
| `--max-samples` | `5000` | `-1` (전체 21,155) | 더 많은 학습 데이터로 일반화 능력 향상 |
| `--epochs` | `1` | `3` | 반복 학습으로 패턴 습득 강화 |
| `--max-length` | `256` | `512` | 긴 답변 생성 능력 확보 |
| `--learning-rate` | `2e-4` | `1e-4` | 안정적인 수렴, 과적합 방지 |

#### 레벨 2: LoRA 강화

```bash
python finetune.py \
  --max-samples -1 \
  --epochs 3 \
  --max-length 512 \
  --learning-rate 1e-4 \
  --lora-r 32 \
  --lora-alpha 64
```

| 변경 | 기본값 | 강화값 | 효과 |
|------|--------|--------|------|
| `--lora-r` | `8` | `32` | 학습 가능 파라미터 4배 증가, 표현력 향상 |
| `--lora-alpha` | `16` | `64` | LoRA 가중치 반영 비율 증가 (alpha/r = 2 유지) |

#### 레벨 3: Full Fine-tuning

SmolLM2-135M은 소형 모델이므로 LoRA 대신 전체 파라미터를 학습하면 품질이 크게 향상됩니다.

```bash
python finetune.py \
  --no-lora \
  --max-samples -1 \
  --epochs 5 \
  --max-length 512 \
  --learning-rate 5e-5 \
  --batch-size 4 \
  --gradient-accumulation-steps 4
```

| 변경 | 기본값 | 강화값 | 효과 |
|------|--------|--------|------|
| `--no-lora` | LoRA ON | Full FT | 전체 파라미터 학습 (135M 전부) |
| `--learning-rate` | `2e-4` | `5e-5` | Full FT는 낮은 학습률 필수 |
| `--epochs` | `1` | `5` | 충분한 반복으로 완전 학습 |

#### 품질에 영향을 미치는 핵심 요소

| 순위 | 요소 | 설명 |
|------|------|------|
| 1 | **데이터 양** | `--max-samples -1`로 전체 데이터 사용 |
| 2 | **에폭 수** | 3~5 에폭 권장 (과적합 주의: loss가 다시 올라가면 중단) |
| 3 | **Full FT vs LoRA** | 소형 모델(~1B 이하)은 Full FT가 유리 |
| 4 | **LoRA rank** | LoRA 사용 시 r=16~32로 올리면 효과적 |
| 5 | **학습률** | 너무 높으면 발산, 너무 낮으면 미학습. LoRA: 1e-4, Full: 5e-5 권장 |
| 6 | **max-length** | 짧으면 긴 답변 절삭. 512 권장 |

> **참고**: SmolLM2-135M은 경량 모델이므로 복잡한 한국어 질문에 대한 답변 품질에 한계가 있습니다. 실무 수준의 품질이 필요하면 더 큰 모델(1B+ 파라미터)을 고려하세요.

### GPU 자동 최적화

GPU가 감지되면 다음 최적화가 자동으로 적용됩니다:

- **FP16/BF16 자동 선택**: GPU가 BF16을 지원하면 BF16, 아니면 FP16
- **torch.compile**: PyTorch 2.0+ 환경에서 자동 활성화
- **pin_memory + prefetch**: 데이터 로딩 파이프라인 가속

---

## 2. ResNet DDP 분산 학습

PyTorch DDP 기반 ResNet-50 분산 학습 예제입니다.  
가짜 데이터셋을 사용하여 GPU 워크로드 테스트에 적합합니다.

### 실행 방법

```bash
# GPU 사용
python train.py --epochs 50 --batch-size 128

# CPU 모드
python train.py --cpu --epochs 10 --batch-size 32
```

자세한 옵션은 `python train.py --help`를 참고하세요.

> **참고**: `train.py`는 `mp.spawn` + `MASTER_ADDR=127.0.0.1` 하드코딩 방식이라 **단일 노드** 멀티 GPU 전용입니다. 멀티 노드 분산학습(AstraGo TrainJob)은 아래 `mnist_ddp.py`를 사용하세요.

---

## 3. MNIST 분산학습 (torchrun / AstraGo TrainJob)

PyTorch DDP 기반 MNIST 분산학습 예제입니다.
**torchrun 컨트랙트**로 작성되어 단일 프로세스 → 단일 노드 멀티 GPU → **멀티 노드**까지 코드 변경 없이 확장됩니다. AstraGo v2의 분산학습(Kubeflow Trainer v2 `TrainJob`) 워크로드 검증에 적합합니다.

### 주요 특징

- **torchrun 기반**: 프로세스 스폰과 분산 좌표(`RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT`) 주입은 torchrun/Trainer 컨트롤러가 담당하고, 스크립트는 환경변수만 읽습니다. `mp.spawn` 미사용.
- **완전 오프라인**: MNIST 데이터(`data/MNIST/raw/*.gz`)를 레포에 번들하고 `.gz`를 직접 읽습니다. 모델은 코드로 정의한 소형 CNN이라 사전학습 가중치 다운로드가 없습니다. **torchvision 불필요** (`torch` + stdlib만).
- **자동 fallback**: 환경변수가 없으면(`python mnist_ddp.py`) 단일 프로세스로 실행됩니다.
- **CPU / 단일 GPU / 멀티 GPU / 멀티 노드 지원**

### 파일

```
├── mnist_ddp.py         # MNIST 분산학습 스크립트
├── data/MNIST/raw/      # 번들된 MNIST (idx-ubyte .gz, ~11MB)
└── trainjob.yaml        # Kubeflow TrainJob 매니페스트 예제
```

### 실행 방법

```bash
# 1) 단일 프로세스 (torchrun 없이)
python mnist_ddp.py --epochs 5

# 2) 단일 노드 N GPU (또는 CPU 프로세스 N개)
torchrun --nproc_per_node=2 --standalone mnist_ddp.py --epochs 5

# 3) CPU 모드
python mnist_ddp.py --cpu --epochs 1
```

### AstraGo TrainJob으로 실행

AstraGo UI로 분산학습 잡을 만들면 이 레포가 git-sync로 파드에 클론되고, 실행 커맨드가 `torchrun --nproc_per_node=<GPU수> mnist_ddp.py ...`로 자동 래핑됩니다. `numNodes`(노드 수) × 노드당 GPU 수 = 전체 world_size.

kubectl로 직접 적용하려면 `trainjob.yaml`을 참고하세요.

```bash
kubectl apply -f trainjob.yaml
```

### 주요 옵션

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--epochs` | 학습 에폭 수 | `5` |
| `--batch-size` | 프로세스당 배치 크기 | `64` |
| `--lr` | 학습률 | `1e-3` |
| `--data-dir` | MNIST 데이터 디렉토리 | `./data` |
| `--save-dir` | 모델 저장 경로 (`OUTPUT_DIR` env 우선) | `./checkpoints` |
| `--num-workers` | DataLoader num_workers | `2` |
| `--cpu` | CPU 강제 실행 | OFF |

모델은 rank 0에서만 저장됩니다(`mnist_cnn.pt`). loss는 매 에폭 `all_reduce`로 전 프로세스 평균을 rank 0이 출력합니다.

---

## 실행 환경

- Python 3.12 이상
- PyTorch 2.0 이상
- CUDA 지원 GPU (권장, CPU도 가능)
- 권장 이미지: `nvcr.io/nvidia/pytorch:24.10-py3`
