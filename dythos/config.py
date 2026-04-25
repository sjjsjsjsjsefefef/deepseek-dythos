import torch
from dataclasses import dataclass, field
from typing import List
from enum import IntEnum


class ThinkingLevel(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    XHIGH = 4
    MAX = 5


@dataclass
class DythosConfig:
    variant: str = "pro"  # "pro" or "flash"

    # Core dimensions
    vocab_size: int = 128256
    d_model: int = 8192
    n_heads: int = 64
    head_dim: int = 512
    n_kv_heads: int = 8

    # Architecture depth
    n_prelude_layers: int = 3
    n_coda_layers: int = 3
    max_loop_depth: int = 64
    loop_hidden_ratio: float = 1.0

    # Thinking levels (auto-populated)
    thinking_level_names: List[str] = field(default_factory=list)
    thinking_level_loops: List[int] = field(default_factory=list)
    thinking_level_halts: List[float] = field(default_factory=list)
    thinking_level_mins: List[int] = field(default_factory=list)

    # MoE
    n_experts: int = 384
    n_active_experts: int = 6
    n_shared_experts: int = 2
    d_ff: int = 2048
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 0.001

    # DSA2 Attention
    use_mla_compression: bool = True
    kv_compression_dim: int = 512
    sliding_window: int = 4096
    sparse_block_size: int = 64
    sparse_n_blocks: int = 16
    use_qk_norm: bool = True

    # Stability
    spectral_radius_cap: float = 0.99
    lti_inject_scale: float = 0.1

    # Training
    pretrain_context: int = 32768
    max_context_len: int = 1_000_000
    use_mixed_precision: bool = True

    # Hierarchical media compression
    video_tokens_per_second: int = 16
    audio_tokens_per_second: int = 8
    video_spatial_size: int = 384
    media_max_hours: float = 2.0

    # Feature toggles (shared across Pro & Fast)
    use_mamba_prelude: bool = True
    use_hypernet: bool = True
    use_episodic_memory: bool = True
    use_latent_verifier: bool = True
    use_constitutional_critic: bool = True
    use_draft_loop: bool = True
    use_mtp: bool = True
    mtp_depth: int = 1
    reflection_loops: int = 8

    # Super-expert fusion sources
    super_expert_sources: List[str] = field(default_factory=lambda: [
        "zai-org/GLM-5.1",
        "moonshotai/Kimi-K2.6",
        "tencent/Hy3-preview",
    ])

    def __post_init__(self):
        if self.variant == "flash":
            self.d_model = 4096
            self.n_heads = 32
            self.n_kv_heads = 4
            self.max_loop_depth = 12
            self.model_name = "DeepSeek Dythos Fast"
            if not self.thinking_level_names:
                self.thinking_level_names = ["none", "thinking"]
                self.thinking_level_loops = [1, 6]
                self.thinking_level_halts = [0.98, 0.75]
                self.thinking_level_mins = [0, 2]
        else:
            self.model_name = "DeepSeek Dythos"
            if not self.thinking_level_names:
                self.thinking_level_names = ["none", "low", "medium", "high", "xhigh", "max"]
                self.thinking_level_loops = [1, 4, 8, 16, 32, 64]
                self.thinking_level_halts = [0.98, 0.80, 0.60, 0.45, 0.30, 0.15]
                self.thinking_level_mins = [0, 2, 4, 8, 16, 32]

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
