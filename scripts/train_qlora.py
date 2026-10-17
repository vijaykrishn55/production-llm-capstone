"""QLoRA supervised fine-tuning. Run only in a CUDA-enabled training environment."""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not (row.get("output") or row.get("messages")):
            raise ValueError(f"Row {line_number} needs output or messages")
        rows.append(row)
    if not rows:
        raise ValueError(f"No training records found in {path}")
    return rows


def to_training_text(row: dict, tokenizer) -> str:
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
    else:
        user = str(row.get("instruction", row.get("prompt", ""))).strip()
        extra = str(row.get("input", "")).strip()
        if extra:
            user = f"{user}\n\nInput:\n{extra}"
        messages = [
            {"role": "system", "content": "You are a helpful AI engineering assistant. Be accurate and concise."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(row["output"]).strip()},
        ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--data", default=os.getenv("TRAIN_DATA", "training/instruction_dataset.jsonl"))
    parser.add_argument("--output", default=os.getenv("ADAPTER_OUTPUT", "artifacts/qlora-adapter"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA requires a CUDA GPU. Run this script on Kaggle or another CUDA runtime, not the API laptop/container.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.getenv("HF_TOKEN"), trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=dtype,
        token=os.getenv("HF_TOKEN"),
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    rows = load_rows(Path(args.data))
    dataset = Dataset.from_list([{"text": to_training_text(row, tokenizer)} for row in rows])
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    train_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"QLoRA adapter saved to: {args.output}")
    print("Review training loss and run a base-versus-adapter quality check before deployment.")


if __name__ == "__main__":
    main()
