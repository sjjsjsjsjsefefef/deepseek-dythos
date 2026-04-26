#!/usr/bin/env python3
"""
Export thinking-level checkpoints.
Usage:
    python export.py --variant pro --output-dir ./exports
"""
import argparse
import os
import json
import torch
from dythos import DythosConfig, DeepSeekDythos


def export_thinking_levels(model, variant, base_dir="./exports"):
    os.makedirs(base_dir, exist_ok=True)
    slug = "dythos" if variant == "pro" else "dythos-fast"
    for level in range(model.thinking_ctrl.n_levels):
        name = model.thinking_ctrl.level_names[level]
        save_dir = os.path.join(base_dir, f"deepseek-{slug}-{name}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        meta = {
            "model_type": "deepseek_dythos_v36",
            "variant": variant,
            "variant_name": model.cfg.model_name,
            "thinking_level": level,
            "thinking_level_name": name,
            "max_loop_depth": int(model.thinking_ctrl.base_loops[level].item()),
            "identity": model.identity_anchor.get_name(0 if variant == "pro" else 1),
            "context_length": 1_000_000,
            "video_tokens_per_second": model.cfg.video_tokens_per_second,
            "audio_tokens_per_second": model.cfg.audio_tokens_per_second,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  💾 {model.cfg.model_name} [{name.upper()}] -> {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export Dythos checkpoints")
    parser.add_argument("--variant", type=str, default="pro", choices=["pro", "flash"])
    parser.add_argument("--output-dir", type=str, default="./exports")
    args = parser.parse_args()

    device = "cuda"
    cfg = DythosConfig(variant=args.variant)
    model = DeepSeekDythos(cfg).to(device=device, dtype=torch.bfloat16)
    export_thinking_levels(model, args.variant, base_dir=args.output_dir)


if __name__ == "__main__":
    main()
