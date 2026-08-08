#!/usr/bin/env python
"""
structural smoke test for the phase 3 cnn. no training - it proves the model
builds, a REAL batch from dataset.npz flows through both heads with the right
shapes, both losses compute and backprop with finite grads, and the TCN config
path also runs. seconds on CPU, catches every shape/wiring bug before Colab.

    python tests/smoke_cnn.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.nets import FlareCNN                            # noqa: E402
from src.losses import focal_loss, weighted_bce          # noqa: E402
from src.data_torch import make_loaders, pos_weights     # noqa: E402

NPZ = ROOT / "data" / "dataset.npz"

torch.manual_seed(0)
train, val, test = make_loaders(NPZ, batch_size=64)
pw = pos_weights(NPZ)
print(f"pos_weight  detection {pw['detection']:.1f}   warning {pw['warning']:.1f}")

xb, yd, yw = next(iter(train))
print(f"batch {tuple(xb.shape)}   positives in batch: det {int(yd.sum())} warn {int(yw.sum())}")

# baseline cnn: dilation 1, no residual
net = FlareCNN()
det, warn = net(xb)
assert det.shape == (xb.shape[0],) and warn.shape == (xb.shape[0],), "head shape wrong"
assert torch.isfinite(det).all() and torch.isfinite(warn).all(), "non-finite logits"

# weighted bce on detection, focal on the brutal warning imbalance
ld = weighted_bce(det, yd, pw["detection"])
lw = focal_loss(warn, yw, alpha=0.25, gamma=2.0)
(ld + lw).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all()
           for p in net.parameters()), "missing / NaN grads"
n_par = sum(p.numel() for p in net.parameters())
print(f"cnn  params {n_par:,}  det_loss {ld.item():.4f}  warn_loss {lw.item():.4f}  grads ok")

# receptive field of this baseline: 1 + n_layers*(k-1) steps at 10s cadence
rf = 1 + 3 * (5 - 1)
print(f"cnn  receptive field ~{rf} steps = {rf*10}s (~{rf*10/60:.1f} min) - short by design")

# same code path as a TCN - proves 'TCN-ready' is real, not a slogan
tcn = FlareCNN(dilations=[1, 2, 4], residual=True)
d2, w2 = tcn(xb)
assert d2.shape == w2.shape == (xb.shape[0],) and torch.isfinite(d2).all()
rf_t = 1 + (5 - 1) * (1 + 2 + 4)
print(f"tcn  dilations [1,2,4] residual on  forward ok  RF ~{rf_t} steps "
      f"({rf_t*10/60:.1f} min, covers the rise)")

print("\nsmoke passed: model + both losses + backward + tcn path all green on real data.")
