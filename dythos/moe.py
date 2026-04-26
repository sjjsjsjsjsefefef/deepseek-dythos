import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .utils import Expert, HyperConnection
from .config import DythosConfig


class SuperExpertMoE(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])
        self.shared_experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_shared_experts)])
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.hyper = HyperConnection(cfg.d_model)
        self._inject_super_experts()

    def _inject_super_experts(self):
        sources = self.cfg.super_expert_sources
        if not sources:
            return
        print(f"[SuperExpert] Fusing {len(sources)} external experts...")
        super_exp_idx = [0, 1, 2]
        for idx, src_id in enumerate(sources):
            try:
                ext = AutoModel.from_pretrained(src_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
                src_w1 = ext.layers[0].mlp.experts[0].w1.weight.data
                src_w2 = ext.layers[0].mlp.experts[0].w2.weight.data
                target = self.experts[super_exp_idx[idx]]
                with torch.no_grad():
                    t_w1 = target.w1.weight
                    t_w2 = target.w2.weight
                    t_w1.copy_(0.5 * t_w1 + 0.5 * src_w1[:t_w1.size(0), :t_w1.size(1)])
                    t_w2.copy_(0.5 * t_w2 + 0.5 * src_w2[:t_w2.size(0), :t_w2.size(1)])
                print(f"  ✓ {src_id} -> expert #{super_exp_idx[idx]}")
            except Exception as e:
                print(f"  ⚠ {src_id}: {e}")

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        weights, selected = torch.topk(probs, self.cfg.n_active_experts, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        aux_loss = self.cfg.moe_aux_loss_coef * (probs.mean(dim=0).pow(2).sum())
        z_loss = self.cfg.moe_z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (selected == i).any(dim=-1)
            if mask.any():
                exp_in = x_flat[mask]
                exp_out = expert(exp_in)
                w = weights[mask][selected[mask] == i].unsqueeze(-1)
                out[mask] += w * exp_out

        for sh_exp in self.shared_experts:
            out += sh_exp(x_flat)

        return self.hyper(out.view(B, T, C), x), aux_loss + z_loss
