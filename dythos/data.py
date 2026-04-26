from datasets import load_dataset, interleave_datasets, Dataset


def load_reasoning_mix(variant_name: str):
    ds_names = [
        ("eddieran/opus-4.7-reasoning-cot", 0.18),
        ("Roman1111111/claude-opus-4.6-10000x", 0.13),
        ("TeichAI/claude-4.5-opus-high-reasoning-250x", 0.13),
        ("DJLougen/Talos-kimi-k2.6-Hermes-synthetic", 0.13),
        ("Jackrong/GLM-5.1-Reasoning-1M-Cleaned", 0.18),
        ("khazarai/qwen3.6-plus-high-reasoning-500x", 0.15),
    ]
    datasets = []
    for name, weight in ds_names:
        try:
            d = load_dataset(name, split="train", streaming=True)
            datasets.append(d)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ⚠ {name}: {e}")

    mixed = interleave_datasets(datasets, probabilities=[w for _, w in ds_names], seed=42) if datasets else None

    identity_qa = [
        {"text": f"User: What is your name?\nAssistant: I am {variant_name}."},
        {"text": f"User: Who are you?\nAssistant: I am {variant_name}, a recurrent-depth multimodal reasoning model."},
        {"text": f"User: What model is this?\nAssistant: You are speaking to {variant_name}."},
    ]
    identity_ds = Dataset.from_list(identity_qa).shuffle(seed=42)

    if mixed:
        return interleave_datasets([mixed, identity_ds], probabilities=[0.9, 0.1], seed=42)
    return identity_ds
