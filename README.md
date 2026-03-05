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

---

## 실행 환경

- Python 3.12 이상
- PyTorch 2.0 이상
- CUDA 지원 GPU (권장, CPU도 가능)
- 권장 이미지: `nvcr.io/nvidia/pytorch:24.10-py3`
