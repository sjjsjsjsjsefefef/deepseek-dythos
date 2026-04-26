#!/usr/bin/env python3
"""
Text-only inference script.
Usage:
    python inference.py --variant pro --prompt "Explain recurrent loops"
"""
import argparse
import torch
from transformers import AutoTokenizer
from dythos import DythosConfig, DeepSeekDythos, DythosInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Run DeepSeek Dythos inference")
    parser.add_argument("--variant", type=str, default="pro", choices=["pro", "flash"])
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--thinking-level", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    device = "cuda"
    cfg = DythosConfig(variant=args.variant)
    model = DeepSeekDythos(cfg).to(device=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Pro", trust_remote_code=True)

    engine = DythosInferenceEngine(model, tokenizer)
    result = engine.generate(
        [args.prompt],
        thinking_level=args.thinking_level,
        variant_idx=0 if args.variant == "pro" else 1,
        max_new_tokens=args.max_tokens,
    )
    print("\n📝 Output:\n", result[0])


if __name__ == "__main__":
    main()
