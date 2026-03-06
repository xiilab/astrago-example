import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "packages"))

import argparse
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


PROMPT_TEMPLATE = "### 질문: {instruction}\n\n### 답변: {output}"
PROMPT_NO_INPUT = "### 질문: {instruction}\n\n### 답변: "

DEFAULT_MODEL = "./model"
DEFAULT_DATASET = "./dataset"


def format_example(example: dict) -> str:
    output = example.get("output", "")
    if output:
        return PROMPT_TEMPLATE.format(
            instruction=example["instruction"],
            output=output,
        )
    return PROMPT_NO_INPUT.format(instruction=example["instruction"])


def tokenize_fn(examples, tokenizer, max_length):
    texts = [format_example({"instruction": inst, "output": out})
             for inst, out in zip(examples["instruction"], examples["output"])]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def apply_lora(model, r: int, alpha: int, dropout: float, target_modules: list[str] | None = None):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_koalpaca(dataset_path: str, max_samples: int | None = None):
    if os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path, split="train")

    if max_samples and max_samples > 0 and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    return dataset


def main():
    parser = argparse.ArgumentParser(description="KoAlpaca 파인튜닝 (SmolLM2-135M + LoRA)")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL,
                        help=f"모델 경로 또는 HuggingFace 모델명 (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET,
                        help=f"데이터셋 경로 또는 HuggingFace 데이터셋명 (default: {DEFAULT_DATASET})")
    parser.add_argument("--output-dir", type=str, default="./finetuned-model",
                        help="파인튜닝 결과 저장 경로 (default: ./finetuned-model)")
    parser.add_argument("--epochs", type=int, default=1, help="학습 에폭 수 (default: 1)")
    parser.add_argument("--batch-size", type=int, default=8, help="배치 크기 (default: 8)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2,
                        help="그래디언트 누적 스텝 (default: 2)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="학습률 (default: 2e-4)")
    parser.add_argument("--max-length", type=int, default=256, help="최대 토큰 길이 (default: 256)")
    parser.add_argument("--warmup-steps", type=int, default=50, help="워밍업 스텝 (default: 50)")
    parser.add_argument("--logging-steps", type=int, default=10, help="로깅 간격 (default: 10)")
    parser.add_argument("--save-steps", type=int, default=500, help="체크포인트 저장 간격 (default: 500)")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="최대 학습 샘플 수 (default: 5000, 전체: -1)")

    # LoRA
    parser.add_argument("--use-lora", action="store_true", default=True, help="LoRA 사용 (default: True)")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false", help="LoRA 비활성화 (full fine-tuning)")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    parser.add_argument("--lora-target-modules", type=str, nargs="+", default=None,
                        help="LoRA 적용 대상 모듈 (default: q/k/v/o_proj, gate/up/down_proj)")

    parser.add_argument("--fp16", action="store_true", help="FP16 학습 활성화")
    parser.add_argument("--bf16", action="store_true", help="BF16 학습 활성화")
    parser.add_argument("--cpu", action="store_true", help="CPU 모드로 강제 실행")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(script_dir, args.model_path)
    if not os.path.isabs(args.dataset_path):
        args.dataset_path = os.path.join(script_dir, args.dataset_path)
    args.model_path = os.path.realpath(args.model_path)
    args.dataset_path = os.path.realpath(args.dataset_path)

    # --- 디바이스 확인 ---
    if args.cpu:
        device_info = "CPU (강제)"
    elif torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        device_info = f"GPU x{gpu_count} ({gpu_name})"
    else:
        device_info = "CPU (GPU 없음)"
        args.cpu = True

    print("=" * 60)
    print("  KoAlpaca 파인튜닝")
    print("=" * 60)
    print(f"  모델       : {args.model_path}")
    print(f"  데이터셋   : {args.dataset_path}")
    print(f"  디바이스   : {device_info}")
    print(f"  LoRA       : {'ON' if args.use_lora else 'OFF (Full Fine-tuning)'}")
    print(f"  에폭       : {args.epochs}")
    print(f"  배치 크기  : {args.batch_size} x {args.gradient_accumulation_steps} (accumulation)")
    print(f"  학습률     : {args.learning_rate}")
    print(f"  최대 길이  : {args.max_length}")
    print("=" * 60)

    # --- 모델 & 토크나이저 로드 ---
    print("\n[1/4] 모델 및 토크나이저 로딩 중...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    if args.use_lora:
        print("[1/4] LoRA 어댑터 적용 중...")
        model = apply_lora(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )

    # --- 데이터셋 로드 & 토크나이즈 ---
    print("\n[2/4] 데이터셋 로딩 중...")
    dataset = load_koalpaca(args.dataset_path, args.max_samples)
    print(f"  학습 샘플 수: {len(dataset)}")

    print("[2/4] 토크나이징 중...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_fn(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --- 학습 설정 ---
    print("\n[3/4] 학습 시작...")
    use_cpu = args.cpu
    use_gpu = not use_cpu and torch.cuda.is_available()

    fp16 = args.fp16
    bf16 = args.bf16
    gpu_major = 0
    if use_gpu:
        gpu_major, _ = torch.cuda.get_device_capability(0)
        if not fp16 and not bf16:
            if gpu_major >= 8:
                bf16 = True
            else:
                fp16 = True

    use_torch_compile = use_gpu and gpu_major >= 8 and hasattr(torch, "compile")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=fp16,
        bf16=bf16,
        use_cpu=use_cpu,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=use_gpu,
        dataloader_prefetch_factor=2 if use_gpu else None,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        torch_compile=use_torch_compile,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # --- 모델 저장 ---
    print("\n[4/4] 모델 저장 중...")
    if args.use_lora:
        print("  LoRA 어댑터를 기본 모델에 병합 중...")
        model = model.merge_and_unload()
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)

    import json
    import shutil
    tokenizer_files = ["tokenizer.json", "special_tokens_map.json", "merges.txt", "vocab.json"]
    for f in tokenizer_files:
        src = os.path.join(args.model_path, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(final_dir, f))

    tc_src = os.path.join(args.model_path, "tokenizer_config.json")
    if os.path.isfile(tc_src):
        with open(tc_src) as f:
            tc = json.load(f)
        tc.pop("extra_special_tokens", None)
        tc.pop("is_local", None)
        with open(os.path.join(final_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tc, f, indent=2)
    print(f"  모델 저장 완료: {final_dir}")

    # --- 간단한 생성 테스트 ---
    print("\n" + "=" * 60)
    print("  생성 테스트")
    print("=" * 60)

    model.eval()
    device = next(model.parameters()).device
    test_prompt = "### 질문: 인공지능이란 무엇인가요?\n\n### 답변: "
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n{generated}")
    print("\n" + "=" * 60)
    print("  파인튜닝 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
