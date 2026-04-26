import torch
import torch.nn as nn
from .utils import LTILinear
from .config import DythosConfig


class ThinkingLevelController(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.level_names = cfg.thinking_level_names
        self.n_levels = len(self.level_names)
        self.prefix_emb = nn.Embedding(self.n_levels, cfg.d_model)

        loops = cfg.thinking_level_loops[:self.n_levels]
        halts = cfg.thinking_level_halts[:self.n_levels]
        mins = cfg.thinking_level_mins[:self.n_levels]

        self.register_buffer("base_loops", torch.tensor(loops, dtype=torch.long))
        self.register_buffer("halt_thresholds", torch.tensor(halts, dtype=torch.float32))
        self.register_buffer("min_loops", torch.tensor(mins, dtype=torch.long))

        self.loop_scales = nn.Parameter(torch.ones(self.n_levels, 1, 1, 1))
        self.halt_biases = nn.Parameter(torch.zeros(self.n_levels, 1, 1))

    def get_prefix(self, level_idx: torch.Tensor):
        if level_idx.dim() == 1:
            level_idx = level_idx.unsqueeze(1)
        return self.prefix_emb(level_idx)

    def get_params(self, level_idx: int):
        return {
            "max_loops": int(self.base_loops[level_idx].item()),
            "halt_threshold": float(self.halt_thresholds[level_idx].item()),
            "min_loops": int(self.min_loops[level_idx].item()),
            "scale": self.loop_scales[level_idx],
            "halt_bias": self.halt_biases[level_idx],
        }


class IdentityAnchor(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.prefix_emb = nn.Embedding(2, cfg.d_model)
        self.names = ["DeepSeek Dythos", "DeepSeek Dythos Fast"]

    def get_prefix(self, variant_idx: int, batch_size: int, device):
        idx = torch.full((batch_size, 1), variant_idx, dtype=torch.long, device=device)
        return self.prefix_emb(idx)

    def get_name(self, variant_idx: int) -> str:
        return self.names[variant_idx]


class EpisodicMemory(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.cfg = cfg
        self.write_gate = LTILinear(cfg.d_model, cfg.kv_compression_dim, cap=cfg.spectral_radius_cap)
        self.read_proj = nn.Linear(cfg.kv_compression_dim, cfg.d_model)
        self._store = {}

    def write(self, session_id, latent, loop_idx):
        if not self.cfg.use_episodic_memory or session_id is None:
            return
        compressed = self.write_gate(latent.mean(dim=1, keepdim=True))
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(compressed.detach().cpu())

    def read(self, session_id, device):
        if not self.cfg.use_episodic_memory or session_id is None or session_id not in self._store:
            return None
        mem = torch.cat(self._store[session_id], dim=1).to(device)
        return self.read_proj(mem.mean(dim=1, keepdim=True))

    def clear(self, session_id):
        self._store.pop(session_id, None)

    def reset(self):
        self._store.clear()


class LatentVerifier(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.value_head = nn.Linear(cfg.d_model, 1)
        self.contradiction_head = nn.Linear(cfg.d_model, 1)
        self.coherence_head = nn.Linear(cfg.d_model, 1)

    def forward(self, latent):
        value = self.value_head(latent)
        contradiction = torch.sigmoid(self.contradiction_head(latent))
        coherence = torch.sigmoid(self.coherence_head(latent))
        return value, contradiction, coherence


class ConstitutionalCritic(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.n_principles = 8
        self.principles = nn.Parameter(torch.randn(self.n_principles, cfg.d_model) * 0.02)
        self.violation_proj = nn.Sequential(
            nn.RMSNorm(cfg.d_model),
            nn.Linear(cfg.d_model, self.n_principles),
        )
        self.reflection_gate = nn.Linear(self.n_principles, 1)

    def forward(self, latent):
        violation_logits = self.violation_proj(latent.mean(dim=1))
        reflection_depth = torch.sigmoid(self.reflection_gate(violation_logits.mean(dim=1)))
        return reflection_depth, violation_logits


class DraftLoopHead(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.RMSNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.max_loop_depth),
        )
        self.confidence = nn.Linear(cfg.d_model, 1)

    def forward(self, prelude_out):
        pooled = prelude_out.mean(dim=1)
        return self.predictor(pooled), torch.sigmoid(self.confidence(pooled))


class LoopHyperNet(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.loop_emb = nn.Embedding(cfg.max_loop_depth, cfg.d_model // 4)
        self.down = nn.Linear(cfg.d_model // 4, cfg.d_model // 16)
        self.up = nn.Linear(cfg.d_model // 16, cfg.d_model)
        self.act = nn.SiLU()

    def forward(self, loop_idx: int):
        z = self.loop_emb(torch.tensor(loop_idx, device=self.loop_emb.weight.device))
        return self.up(self.act(self.down(z)))
