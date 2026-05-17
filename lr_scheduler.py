"""
Noam Learning Rate Scheduler
Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           https://arxiv.org/abs/1706.03762

Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

WHY does the Transformer need this special schedule?
-----------------------------------------------------
A Transformer trained with a *constant* learning rate from step 0 tends to
diverge: at the start the attention weights are essentially random, the
softmax produces near-uniform distributions, and the resulting gradients are
both large and noisy. A big LR on those noisy gradients pushes the weights
into a bad region the model never recovers from.

The Noam schedule fixes this with two phases:
  * WARM-UP  (step < warmup_steps): LR rises *linearly* from ~0. Tiny early
    steps let the attention layers settle before any big update is applied.
  * DECAY    (step > warmup_steps): LR falls like 1/sqrt(step), taking
    smaller and smaller steps as the model approaches a good minimum.

The peak LR occurs exactly at step == warmup_steps.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


# ─────────────────────────────────────────────
#  NoamScheduler
# ─────────────────────────────────────────────

class NoamScheduler(LRScheduler):
    """
    Noam learning rate scheduler as described in "Attention Is All You Need".

    Applies a warm-up phase where LR increases linearly, followed by
    a decay phase where LR decreases proportional to the inverse square
    root of the step number.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        d_model          (int)  : Model dimensionality (embedding size).
        warmup_steps     (int)  : Number of warm-up steps before decay begins.
        last_epoch       (int)  : The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        # Store our two hyper-parameters BEFORE calling super().__init__,
        # because the parent constructor immediately calls get_lr(), which
        # in turn needs self.d_model and self.warmup_steps to exist.
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    # ------------------------------------------------------------------
    def _get_lr_scale(self) -> float:
        """
        Compute the Noam scaling factor for the current step.

        Returns:
            float: The scalar multiplier applied to the base learning rate.
        """
        # PyTorch's scheduler counts steps in `last_epoch`. It starts at 0,
        # so we use step = last_epoch + 1 to make the first real step be 1
        # (the formula has step^(-0.5), which is undefined at step 0).
        step = self.last_epoch + 1

        # Noam formula, split for readability:
        #   term_a = step^(-0.5)                  -> dominates AFTER warm-up
        #   term_b = step * warmup_steps^(-1.5)   -> dominates DURING warm-up
        term_a = step ** (-0.5)
        term_b = step * (self.warmup_steps ** (-1.5))

        scale = (self.d_model ** (-0.5)) * min(term_a, term_b)
        return scale

    # ------------------------------------------------------------------
    def get_lr(self) -> list[float]:
        """
        Compute learning rates for every param group.

        Called internally by PyTorch's scheduler machinery each step.

        Returns:
            list[float]: New learning rate for each param group.
        """
        scale = self._get_lr_scale()
        # base_lrs holds the LR each param group was created with.
        # The effective LR is base_lr * Noam_scale.
        return [base_lr * scale for base_lr in self.base_lrs]


# ──────────────────────────────────────────────────────────────────────
# Helper — do NOT modify
# ──────────────────────────────────────────────────────────────────────

def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> list[float]:
    """
    Simulate the LR trajectory of NoamScheduler for `total_steps` steps.

    Args:
        d_model      (int): Model dimensionality.
        warmup_steps (int): Warm-up steps.
        total_steps  (int): Number of steps to simulate.

    Returns:
        list[float]: LR value at each step (length == total_steps).
    """
    dummy_model = torch.nn.Linear(1, 1)
    optimizer   = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler   = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return history


# ──────────────────────────────────────────────────────────────────────
# Quick visual check — run:  python lr_scheduler.py
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL      = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS  = 20_000

    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--", label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule  (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.show()
