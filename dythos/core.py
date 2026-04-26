import torch.nn as nn
from .attention import DSA2Attention
from .moe import SuperExpertMoE
from .utils import LTILinear
from .config import DythosConfig


class PreludeLayer(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.attn = DSA2Attention(cfg)
        self.moe = SuperExpertMoE(cfg)
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.norm2 = nn.RMSNorm(cfg.d_model)

    def forward(self, x, loop_idx=0):
        h = self.attn(self.norm1(x), loop_idx=loop_idx)
        h, moe_l = self.moe(self.norm2(x))
        return x + h, moe_l


class RecurrentBlock(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = DSA2Attention(cfg)
        self.moe = SuperExpertMoE(cfg)
        self.norm_attn = nn.RMSNorm(cfg.d_model)
        self.norm_moe = nn.RMSNorm(cfg.d_model)
        self.lti_inject = LTILinear(cfg.d_model, cfg.d_model, cap=cfg.spectral_radius_cap)
        self.halt_proj = nn.Linear(cfg.d_model, 1)

        if cfg.use_hypernet:
            from .controller import LoopHyperNet
            self.hypernet = LoopHyperNet(cfg)
        if cfg.use_latent_verifier:
            from .controller import LatentVerifier
            self.verifier = LatentVerifier(cfg)

    def forward(self, x, loop_idx, level_scale, halt_bias, memory_context=None):
        hyper = None
        if self.cfg.use_hypernet:
            hyper = self.hypernet(loop_idx)

        h = self.norm_attn(x)
        h = self.attn(h, loop_idx=loop_idx, hyper_adapter=hyper)
        if memory_context is not None:
            h = h + memory_context
        x = x + h

        h = self.norm_moe(x)
        h, moe_l = self.moe(h)
        x = x + h + self.lti_inject(x) * self.cfg.lti_inject_scale * level_scale

        halt_logit = self.halt_proj(x.mean(dim=-1, keepdim=True)) + halt_bias

        v = c = co = None
        if self.cfg.use_latent_verifier:
            v, c, co = self.verifier(x)

        return x, halt_logit, moe_l, v, c, co


class CodaLayer(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.RMSNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x):
        return x + self.mlp(x)


class MTPModule(nn.Module):
    def __init__(self, cfg: DythosConfig, lm_head: nn.Linear):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.GELU(),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
        )
        self.proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.proj.weight = lm_head.weight

    def forward(self, x):
        return self.proj(self.mlp(self.norm(x)))
