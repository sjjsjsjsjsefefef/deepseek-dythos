import torch
import torch.nn.functional as F
from typing import Optional, List


class DythosInferenceEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        video_tokens=None,
        audio_tokens=None,
        thinking_level: int = 0,
        variant_idx: int = 0,
        session_id: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.cfg.get_device())
        input_ids = inputs.input_ids
        for _ in range(max_new_tokens):
            logits, _ = self.model(
                input_ids,
                video_tokens=video_tokens,
                audio_tokens=audio_tokens,
                thinking_level=thinking_level,
                variant_idx=variant_idx,
                session_id=session_id,
            )
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            if next_tok.item() == self.tokenizer.eos_token_id:
                break
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def auto_mode(
        self,
        prompt: str,
        video_tokens=None,
        audio_tokens=None,
        variant_idx: int = 0,
        session_id: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.cfg.get_device())
        with torch.no_grad():
            out = self.model(
                inputs.input_ids,
                video_tokens=video_tokens,
                audio_tokens=audio_tokens,
                thinking_level=0,
                variant_idx=variant_idx,
                return_all=True,
            )
            contra = 0.5
            proposed = self.model.thinking_ctrl.n_levels - 1 if contra > 0.3 else max(0, self.model.thinking_ctrl.n_levels - 2)
        return self.generate(
            [prompt],
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            thinking_level=proposed,
            variant_idx=variant_idx,
            session_id=session_id,
            max_new_tokens=max_new_tokens,
        )[0]


class DythosEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def test_identity(self, variant_idx=0):
        print("\n🆔 [Identity]")
        expected = self.model.identity_anchor.get_name(variant_idx)
        prompts = ["What is your name?", "Who are you?", "What model is this?", "Identify yourself."]
        passed = 0
        max_lvl = self.model.thinking_ctrl.n_levels - 1
        for p in prompts:
            inputs = self.tokenizer(p, return_tensors="pt").to(self.model.cfg.get_device())
            with torch.no_grad():
                out = self.model.generate(inputs.input_ids, max_new_tokens=20, thinking_level=max_lvl, variant_idx=variant_idx)
            ans = self.tokenizer.decode(out[0], skip_special_tokens=True)
            if expected.lower() in ans.lower():
                passed += 1
                print(f"  ✅ {ans.strip()}")
            else:
                print(f"  ❌ {ans.strip()} (expected {expected})")
        return passed == len(prompts)

    def test_spectral_stability(self):
        print("\n🔬 [LTI]")
        from .utils import LTILinear
        violations = 0
        for name, m in self.model.named_modules():
            if isinstance(m, LTILinear):
                if m.spectral_norm(m.weight).item() >= m.cap:
                    violations += 1
        if violations == 0:
            print("  ✅ Stable")
        return violations == 0

    def test_multihop(self):
        print("\n🔄 [Multi-Hop]")
        prompt = "Prove recurrent-depth transformers converge when spectral radius < 1."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.cfg.get_device())
        max_lvl = self.model.thinking_ctrl.n_levels - 1
        prev = None
        deltas = []
        max_loops = int(self.model.thinking_ctrl.base_loops[max_lvl].item())
        points = [p for p in [1, 2, 4, 8, 16, 32, 64] if p <= max_loops] or [1, max_loops]
        for d in points:
            with torch.no_grad():
                out = self.model(inputs.input_ids, thinking_level=max_lvl, force_max_loops=d)
            logits = out[0][0, -1, :1000]
            if prev is not None:
                deltas.append((logits - prev).abs().mean().item())
                print(f"  Loops={d:2d} Δ={deltas[-1]:.6f}")
            prev = logits
        ok = len(deltas) >= 2 and deltas[-1] < deltas[0] * 0.5
        print(f"  {'✅' if ok else '⚠'} Converges" if ok else " Diverges?")
        return ok

    def run_full_suite(self, variant_idx=0):
        print(f"\n{'='*60}\n  {self.model.cfg.model_name}\n{'='*60}")
        r = {
            "identity": self.test_identity(variant_idx),
            "lti": self.test_spectral_stability(),
            "multihop": self.test_multihop(),
        }
        print(f"\n📊 {sum(r.values())}/{len(r)} passed.")
        return r
