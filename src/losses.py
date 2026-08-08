# losses for the 88:1 warning imbalance. plain BCE just learns to say "no"
# forever and scores great on accuracy - useless. either reweight the positive
# or focal-down the easy negatives. both take raw logits, not probabilities.

import torch
import torch.nn.functional as F


def weighted_bce(logits, targets, pos_weight):
    # pos_weight ~ neg/pos, so ~88 for warning, ~2-3 for detection. scalar ok.
    pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pw)


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    # scales each example by (1 - p_t)^gamma so confident-correct ones stop
    # dominating the gradient. gamma=2 is the paper default; alpha tilts toward
    # the rare positive. they interact, so tune both on val, not by eye.
    targets = targets.float()
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    if alpha is not None:
        a_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = a_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss   # 'none' -> per-element, for masking gaps later
