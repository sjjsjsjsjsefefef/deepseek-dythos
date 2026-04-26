import torch
import torch.nn as nn
from typing import Optional
from .config import DythosConfig, ThinkingLevel
from .utils import MambaPrelude
from .core import PreludeLayer, RecurrentBlock, CodaLayer, MTPModule
from .controller import (
    ThinkingLevelController,
    IdentityAnchor,
    EpisodicMemory,
    ConstitutionalCritic,
    DraftLoopHead,
)
from .multimodal import HierarchicalVideoEncoder, HierarchicalAudioEncoder


class DeepSeekDythos(nn.Module):
    def __init__(self, cfg: DythosConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.thinking_ctrl = ThinkingLevelController(cfg)
        self.identity_anchor = IdentityAnchor(cfg)
        self.memory = EpisodicMemory(cfg)

        self.prelude = nn.ModuleList()
        if cfg.use_mamba_prelude:
            self.prelude.append(MambaPrelude(cfg))
        for _ in range(cfg.n_prelude_layers - (1 if cfg.use_mamba_prelude else 0)):
            self.prelude.append(PreludeLayer(cfg))

        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList([CodaLayer(cfg) for _ in range(cfg.n_coda_layers)])

        if cfg.use_constitutional_critic:
            self.constitutional_critic = ConstitutionalCritic(cfg)
        if cfg.use_draft_loop:
            self.draft_head = DraftLoopHead(cfg)

        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.use_mtp:
            self.mtp_modules = nn.ModuleList([MTPModule(cfg, self.lm_head) for _ in range(cfg.mtp_depth)])

        self.video_encoder = HierarchicalVideoEncoder(cfg)
        self.audio_encoder = HierarchicalAudioEncoder(cfg)

        self.lm_head.weight = self.embed.weight
        self.post_init()

    def post_init(self):
        from .utils import LTILinear
        for m in self.modules():
            if isinstance(m, LTILinear):
                _ = m.get_stable_weight()

    def forward(
        self,
        input_ids: torch.Tensor,
        video_tokens: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None,
        thinking_level: int = 0,
        variant_idx: int = 0,
        session_id: Optional[str] = None,
        return_halting: bool = False,
        force_max_loops: Optional[int] = None,
        return_all: bool = False,
    ):
        B, T = input_ids.shape
        device = input_ids.device

        if thinking_level >= self.thinking_ctrl.n_levels:
            thinking_level = self.thinking_ctrl.n_levels - 1

        think_prefix = self.thinking_ctrl.get_prefix(
            torch.full((B, 1), thinking_level, dtype=torch.long, device=device)
        )
        id_prefix = self.identity_anchor.get_prefix(variant_idx, B, device)

        x = self.embed(input_ids)
        x = torch.cat([think_prefix, id_prefix, x], dim=1)
        prefix_len = 2

        if video_tokens is not None:
            x = torch.cat([x[:, :prefix_len], video_tokens, x[:, prefix_len:]], dim=1)
            prefix_len += video_tokens.size(1)
        if audio_tokens is not None:
            x = torch.cat([x[:, :prefix_len], audio_tokens, x[:, prefix_len:]], dim=1)
            prefix_len += audio_tokens.size(1)

        moe_loss_acc = 0.0
        for layer in self.prelude:
            if isinstance(layer, MambaPrelude):
                x = layer(x)
            else:
                x, moe_l = layer(x)
                moe_loss_acc += moe_l

        draft_max = None
        if self.cfg.use_draft_loop and not self.training:
            draft_logits, draft_conf = self.draft_head(x)
            draft_max = draft_logits.argmax(dim=-1).max().item() + 2

        memory_context = self.memory.read(session_id, device)

        params = self.thinking_ctrl.get_params(thinking_level)
        max_loops = force_max_loops or params["max_loops"]
        if draft_max is not None:
            max_loops = min(max_loops, draft_max)
        min_loops = params["min_loops"]
        halt_thresh = params["halt_threshold"]
        level_scale = params["scale"]
        halt_bias = params["halt_bias"]

        loop_halts = []
        for i in range(max_loops):
            x_new, halt_logit, moe_l, value, contradiction, coherence = self.recurrent(
                x, i, level_scale, halt_bias, memory_context
            )
            moe_loss_acc += moe_l
            loop_halts.append(halt_logit)
            self.memory.write(session_id, x_new, i)

            if not self.training and i >= min_loops:
                if torch.sigmoid(halt_logit).mean() > halt_thresh:
                    break
            x = x_new

        if self.cfg.use_constitutional_critic and not self.training:
            reflection_depth, _ = self.constitutional_critic(x)
            extra = int(reflection_depth.mean().item() * self.cfg.reflection_loops)
            for j in range(extra):
                x_new, _, moe_l, _, _, _ = self.recurrent(
                    x, max_loops + j, level_scale, halt_bias, memory_context
                )
                moe_loss_acc += moe_l
                x = x_new

        for layer in self.coda:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        mtp_logits = []
        if self.cfg.use_mtp and self.training:
            for mtp in self.mtp_modules:
                mtp_logits.append(mtp(x))

        if return_all:
            return {
                "logits": logits,
                "moe_loss": moe_loss_acc,
                "halts": torch.stack(loop_halts, dim=1) if loop_halts else None,
                "mtp_logits": mtp_logits,
            }
        if return_halting:
            return logits, torch.stack(loop_halts, dim=1), moe_loss_acc
        return logits, moe_loss_acc
