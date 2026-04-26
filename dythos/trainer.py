import torch
import torch.nn.functional as F
from .model import DeepSeekDythos
from .optimizer import Muon
from .data import load_reasoning_mix


class GRPOTrainer:
    def __init__(self, model, ref_model, optimizer, kl_coef=0.01):
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.kl_coef = kl_coef
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def compute_grpo_loss(self, input_ids, attention_mask, rewards, thinking_level, variant_idx=0):
        B = input_ids.size(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model(input_ids, thinking_level=thinking_level, variant_idx=variant_idx, return_all=True)
            logits = out["logits"]
            moe_loss = out["moe_loss"]

            log_probs = F.log_softmax(logits[:, :-1], dim=-1)
            gathered = torch.gather(log_probs, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            seq_logp = (gathered * attention_mask[:, 1:]).sum(dim=1)

            with torch.no_grad():
                ref_logits, _ = self.ref_model(input_ids, thinking_level=thinking_level, variant_idx=variant_idx)
                ref_log_probs = F.log_softmax(ref_logits[:, :-1], dim=-1)
                ref_gathered = torch.gather(ref_log_probs, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                ref_seq_logp = (ref_gathered * attention_mask[:, 1:]).sum(dim=1)

            kl = (ref_log_probs - log_probs).pow(2).mean()
            baseline = rewards.mean()
            advantage = rewards - baseline
            loss = -(seq_logp * advantage).mean() + self.kl_coef * kl + moe_loss

            if out.get("mtp_logits"):
                for mtp_l in out["mtp_logits"]:
                    mtp_shift = mtp_l[:, :-2]
                    mtp_target = input_ids[:, 2:]
                    loss += 0.2 * F.cross_entropy(
                        mtp_shift.reshape(-1, mtp_shift.size(-1)),
                        mtp_target.reshape(-1),
                    )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), kl.item()


def train_dythos(variant="pro", steps=1000, tokenizer=None):
    import os
    device = "cuda"
    from .config import DythosConfig
    cfg = DythosConfig(variant=variant)
    model = DeepSeekDythos(cfg).to(device=device, dtype=torch.bfloat16)
    ref_model = DeepSeekDythos(cfg).to(device=device, dtype=torch.bfloat16)
    ref_model.load_state_dict(model.state_dict())

    muon_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and "embed" not in n and "prefix_emb" not in n]
    adam_params = [p for n, p in model.named_parameters() if p.dim() < 2 or "embed" in n or "prefix_emb" in n or "halt_biases" in n]

    opt_muon = Muon(muon_params, lr=1e-4)
    opt_adam = torch.optim.AdamW(adam_params, lr=1e-4, betas=(0.9, 0.95))

    class DualOpt:
        def zero_grad(self):
            opt_muon.zero_grad()
            opt_adam.zero_grad()
        def step(self):
            opt_muon.step()
            opt_adam.step()

    trainer = GRPOTrainer(model, ref_model, DualOpt(), kl_coef=0.01)
    dataset = load_reasoning_mix(model.cfg.model_name)
    iter_ds = iter(dataset)
    n_levels = model.thinking_ctrl.n_levels

    model.train()
    for step in range(steps):
        level = int(torch.randint(0, n_levels, (1,)).item())
        level_name = model.thinking_ctrl.level_names[level]

        batch = [next(iter_ds) for _ in range(4)]
        texts = [b["text"] for b in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.pretrain_context).to(device)
        rewards = torch.randn(len(batch), device=device)

        loss, kl = trainer.compute_grpo_loss(enc.input_ids, enc.attention_mask, rewards, thinking_level=level, variant_idx=0 if variant == "pro" else 1)

        if step % 50 == 0:
            print(f"[Step {step}] {variant.upper()} Level={level_name.upper()} Loss={loss:.4f} KL={kl:.4f}")

    save_path = f"./dythos-{variant}-unified-v36"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    print(f"\n💾 Saved to {save_path}")
    return model
