import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import einops
from .utils import LoopAwareRoPE
from .config import DythosConfig


class DSA2Attention(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim ** -0.5

        self.kv_compress = nn.Linear(cfg.d_model, cfg.kv_compression_dim)
        self.kv_decompress_k = nn.Linear(cfg.kv_compression_dim, cfg.n_kv_heads * cfg.head_dim)
        self.kv_decompress_v = nn.Linear(cfg.kv_compression_dim, cfg.n_kv_heads * cfg.head_dim)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim)
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        self.rope = LoopAwareRoPE(cfg.head_dim)
        self.register_buffer("swa_mask", None)

        if cfg.use_qk_norm:
            self.q_norm = nn.RMSNorm(cfg.head_dim)
            self.k_norm = nn.RMSNorm(cfg.head_dim)

    def forward(self, x: torch.Tensor, loop_idx: int = 0, hyper_adapter: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        kv_latent = self.kv_compress(x)
        k = self.kv_decompress_k(kv_latent).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.kv_decompress_v(kv_latent).view(B, T, self.n_kv_heads, self.head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)

        q = einops.rearrange(q, 'b t h d -> b h t d')
        k = einops.rearrange(k, 'b t h d -> b h t d')

        if self.cfg.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope(q, T, loop_idx)
        k = self.rope(k, T, loop_idx)

        k = einops.repeat(k, 'b h t d -> b (h g) t d', g=self.n_heads // self.n_kv_heads)
        v = einops.repeat(v, 'b h t d -> b (h g) t d', g=self.n_heads // self.n_kv_heads)

        if self.swa_mask is None or self.swa_mask.size(-1) < T:
            mask = torch.full((T, T), float('-inf'), device=x.device)
            for i in range(T):
                start = max(0, i - self.cfg.sliding_window)
                mask[i, start:i+1] = 0.0
            self.swa_mask = mask

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + self.swa_mask[:T, :T]

        if T > self.cfg.sliding_window:
            block_size = self.cfg.sparse_block_size
            n_blocks = T // block_size
            if n_blocks > 0:
                k_blocks = k[:, :, :n_blocks*block_size, :].reshape(
                    B, self.n_heads, n_blocks, block_size, self.head_dim
                ).mean(dim=3)
                block_scores = torch.matmul(q, k_blocks.transpose(-2, -1)) * self.scale
                topk = min(self.cfg.sparse_n_blocks, n_blocks)
                top_scores, top_idx = block_scores.topk(topk, dim=-1)
                sparse_attn = torch.zeros_like(scores)
                sparse_attn.scatter_(-1, top_idx.repeat(1,1,1,T//n_blocks)[:,:,:,:T],
                                     top_scores.unsqueeze(-1).expand(-1,-1,-1,T//n_blocks)[:,:,:,:T])
                scores = scores + sparse_attn

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h t d -> b t (h d)')
        out = self.out_proj(out)

        if hyper_adapter is not None:
            out = out * (1 + hyper_adapter.unsqueeze(1))
        return out
