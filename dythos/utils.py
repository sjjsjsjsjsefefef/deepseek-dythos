import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class LTILinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, cap: float = 0.99):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if out_features == in_features else None
        self.cap = cap

    def spectral_norm(self, W: torch.Tensor, n_iter: int = 4) -> torch.Tensor:
        u = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
        for _ in range(n_iter):
            v = F.linear(u, W)
            v = v / (v.norm() + 1e-12)
            u = F.linear(v, W.t())
            u = u / (u.norm() + 1e-12)
        sigma = (F.linear(u, W) * v).sum()
        return sigma.abs()

    def get_stable_weight(self):
        sigma = self.spectral_norm(self.weight)
        scale = torch.clamp(self.cap / (sigma + 1e-6), max=1.0)
        return self.weight * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.get_stable_weight(), self.bias)


class LoopAwareRoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 131072, max_loops: int = 128, base: float = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.max_loops = max_loops

    def forward(self, x: torch.Tensor, seq_len: int, loop_idx: int = 0):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        t = t + loop_idx * (self.max_seq_len // self.max_loops)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)[None, None, :, :]
        cos, sin = emb.cos(), emb.sin()
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot_x = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + rot_x * sin


class HyperConnection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, residual):
        return self.alpha * x + self.beta * residual


class Expert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class MambaPrelude(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if HAS_MAMBA and cfg.use_mamba_prelude:
            self.mamba = Mamba2(d_model=cfg.d_model, d_state=64, d_conv=4, expand=2)
            self.is_mamba = True
        else:
            self.gate = nn.Linear(cfg.d_model, cfg.d_model)
            self.conv = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=4, padding=3, groups=cfg.d_model)
            self.out = nn.Linear(cfg.d_model, cfg.d_model)
            self.norm = nn.RMSNorm(cfg.d_model)
            self.is_mamba = False

    def forward(self, x):
        if self.is_mamba:
            return self.mamba(x)
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x_t = x.transpose(1, 2)
        c = self.conv(x_t)[:, :, :x.size(1)].transpose(1, 2)
        return self.out(c * g)
