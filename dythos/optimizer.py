import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, ns_steps=5, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                buf = state["momentum"]
                buf.mul_(group["momentum"]).add_(grad, alpha=1 - group["momentum"])
                G = buf.view(p.shape[0], -1) if p.dim() > 1 else buf.unsqueeze(1)
                for _ in range(group["ns_steps"]):
                    G = (3.0 * G - G @ G.T @ G) * 0.5
                p.add_(G.view_as(p), alpha=-group["lr"])
