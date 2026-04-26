import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import DythosConfig
from .utils import HAS_MAMBA, MambaPrelude


class PerceiverSecondTokenizer(nn.Module):
    def __init__(self, d_model: int, num_latents: int):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, kv: torch.Tensor):
        B = kv.size(0)
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross_attn(self.norm_q(q), self.norm_kv(kv), kv)
        return out + self.mlp(self.norm_out(out))


class HierarchicalVideoEncoder(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(3, cfg.d_model, kernel_size=14, stride=14)
        self.second_tokenizer = PerceiverSecondTokenizer(cfg.d_model, cfg.video_tokens_per_second)
        if cfg.use_mamba_prelude and HAS_MAMBA:
            self.temporal_agg = Mamba2(d_model=cfg.d_model, d_state=64, d_conv=4, expand=2)
        else:
            self.temporal_agg = nn.TransformerEncoderLayer(
                cfg.d_model, 8, cfg.d_model * 2, batch_first=True
            )

    def encode_second(self, frames: torch.Tensor):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        patches = self.patch_embed(frames)
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches.view(B, T * patches.size(1), self.cfg.d_model)
        return self.second_tokenizer(patches)

    def forward(self, video_input):
        if isinstance(video_input, torch.Tensor) and video_input.dim() == 3:
            x = video_input
        else:
            second_tokens = [self.encode_second(sec) for sec in video_input]
            x = torch.cat(second_tokens, dim=1)
        if x.size(1) > 1:
            x = self.temporal_agg(x)
        return x


class HierarchicalAudioEncoder(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.cfg = cfg
        self.mel_conv = nn.Sequential(
            nn.Conv1d(128, cfg.d_model, 25, stride=10, padding=12),
            nn.SiLU(),
            nn.Conv1d(cfg.d_model, cfg.d_model, 25, stride=10, padding=12),
            nn.SiLU(),
        )
        self.second_tokenizer = PerceiverSecondTokenizer(cfg.d_model, cfg.audio_tokens_per_second)

    def encode_second(self, log_mel: torch.Tensor):
        x = log_mel.transpose(1, 2)
        x = self.mel_conv(x)
        x = x.transpose(1, 2)
        return self.second_tokenizer(x)

    def forward(self, audio_input):
        if isinstance(audio_input, torch.Tensor) and audio_input.dim() == 3:
            return audio_input
        second_tokens = [self.encode_second(sec) for sec in audio_input]
        return torch.cat(second_tokens, dim=1)
