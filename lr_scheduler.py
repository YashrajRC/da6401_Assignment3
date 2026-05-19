"""
lr_scheduler.py — Noam Learning Rate Scheduler
Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           https://arxiv.org/abs/1706.03762

Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

WHY the Transformer needs this schedule
----------------------------------------
With a constant LR from step 0, the early attention weights are essentially
random, producing near-uniform softmax outputs and noisy gradients. A large
LR applied to those gradients pushes weights into a bad basin the model
never escapes. The Noam schedule has two phases:

  WARM-UP  (step <= warmup_steps): LR rises linearly from ~0.
    - Keeps updates tiny while the attention layers find a reasonable
      initial direction, preventing early divergence in the softmax layers.

  DECAY    (step >  warmup_steps): LR falls ∝ 1/√step.
    - Progressively smaller steps as the model approaches convergence.

The peak LR occurs exactly at step == warmup_steps.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    """
    Noam learning rate scheduler as described in "Attention Is All You Need".

    Args:
        optimizer    (Optimizer): Wrapped optimizer.
        d_model      (int)      : Model dimensionality (embedding size).
        warmup_steps (int)      : Steps before LR starts decaying.
        last_epoch   (int)      : Index of the last epoch (default -1).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        # Store BEFORE super().__init__ because the parent constructor
        # calls get_lr() immediately, which needs these attributes.
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def _get_lr_scale(self) -> float:
        """
        Noam scaling factor for the current step.

        step = last_epoch + 1  (avoids step=0 where step^-0.5 is undefined).
        scale = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        """
        step  = self.last_epoch + 1
        term_a = step ** (-0.5)                         # decay phase
        term_b = step * (self.warmup_steps ** (-1.5))   # warm-up phase
        return (self.d_model ** (-0.5)) * min(term_a, term_b)

    def get_lr(self) -> list:
        """Return new LR for each param group (called by PyTorch internally)."""
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


# ──────────────────────────────────────────────────────────────────────
# Helper — do NOT modify (used by autograder tests)
# ──────────────────────────────────────────────────────────────────────

def get_lr_history(d_model: int, warmup_steps: int, total_steps: int) -> list:
    """Simulate the LR trajectory for total_steps steps."""
    dummy = torch.nn.Linear(1, 1)
    opt   = optim.Adam(dummy.parameters(), lr=1.0)
    sched = NoamScheduler(opt, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return history


# ──────────────────────────────────────────────────────────────────────
# Quick visual check:  python lr_scheduler.py
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL      = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS  = 20_000

    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--",
                label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule  (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.show()