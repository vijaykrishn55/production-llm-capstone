"""Merge a trained PEFT adapter into its base checkpoint for post-training quantization."""
import argparse
import os
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=os.getenv("ADAPTER_OUTPUT", "artifacts/qlora-adapter"))
    parser.add_argument("--output", default=os.getenv("MERGED_OUTPUT", "artifacts/merged-model"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        token=os.getenv("HF_TOKEN"),
        trust_remote_code=False,
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, token=os.getenv("HF_TOKEN"), trust_remote_code=False)
    tokenizer.save_pretrained(output)
    print(f"Merged deployment checkpoint saved to {output}")


if __name__ == "__main__":
    main()
