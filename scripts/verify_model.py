"""Inspect model artifact metadata and record disk size before deployment."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Merged or quantized HF model directory")
    args = parser.parse_args()
    root = Path(args.model_dir)
    if not root.is_dir():
        raise SystemExit(f"Model directory does not exist: {root}")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise SystemExit("config.json is missing; the directory is not a complete Transformers model")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    files = [p for p in root.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    report = {
        "model_dir": str(root.resolve()),
        "file_count": len(files),
        "disk_size_gib": round(total_bytes / 1024**3, 3),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "quantization_config": config.get("quantization_config"),
        "quantization_metadata_present": bool(config.get("quantization_config")),
        "has_tokenizer": any((root / name).exists() for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")),
        "note": "Metadata inspection only. Confirm quantization by loading with vLLM and benchmarking task quality, latency and VRAM.",
    }
    print(json.dumps(report, indent=2))
    if not report["has_tokenizer"]:
        raise SystemExit("Tokenizer files missing")


if __name__ == "__main__":
    main()
