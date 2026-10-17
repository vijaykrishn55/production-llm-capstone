"""Quantize a merged HF model to AWQ W4A16 using vLLM's LLM Compressor."""
import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("MERGED_OUTPUT", "artifacts/merged-model"))
    parser.add_argument("--output", default=os.getenv("AWQ_OUTPUT", "models/merged-awq"))
    parser.add_argument("--calibration-data", default="training/instruction_dataset.jsonl")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("AWQ quantization requires a supported GPU. Run this on a CUDA notebook/runtime.")
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=args.calibration_data, split="train")
    if "instruction" in dataset.column_names:
        dataset = dataset.map(lambda row: {"text": f"Instruction: {row.get('instruction', '')}\nInput: {row.get('input', '')}\nAnswer: {row.get('output', '')}"})
    if "text" not in dataset.column_names:
        raise SystemExit("Calibration file must contain instruction/output columns or a text column")
    dataset = dataset.select(range(min(args.samples, len(dataset))))
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    recipe = [
        AWQModifier(),
        QuantizationModifier(scheme="W4A16_ASYM", targets=["Linear"], ignore=["lm_head"]),
    ]
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(dataset),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, save_compressed=True, safe_serialization=True)
    tokenizer.save_pretrained(output)
    print(f"AWQ checkpoint saved to {output}; inspect model config and run inference parity tests before release.")


if __name__ == "__main__":
    main()
