# dataset.npz -> torch loaders. X is already scaled (train-fit) and windowed to
# (N,180,6) at 10s, so this just wraps and splits. now takes optional masks so
# walk-forward can pass its own folds; None keeps the fixed chronological split.
# accepts a path, an npz, or a materialized dict (dict avoids reloading X from
# disk every call - matters when you fit many folds/seeds in a loop).

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def as_arrays(npz):
    if isinstance(npz, (str, Path)):
        return np.load(npz, allow_pickle=False)
    return npz                                  # npz obj or plain dict, both index by key


def _subset(z, mask):
    X = torch.from_numpy(z["X"][mask])                      # (n,180,6) float32
    yd = torch.from_numpy(z["y_progress"][mask].astype("float32"))
    yw = torch.from_numpy(z["y_warn"][mask].astype("float32"))
    return TensorDataset(X, yd, yw)


def make_loaders(npz, masks=None, batch_size=256, num_workers=0):
    z = as_arrays(npz)
    if masks is None:
        masks = (z["is_train"], z["is_val"], z["is_test"])
    tr, va, te = (_subset(z, m) for m in masks)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers)
    return mk(tr, True), mk(va, False), mk(te, False)


def pos_weights(npz, train_mask=None):
    # neg/pos on the given train mask, per task. warning is the ~88:1 monster.
    z = as_arrays(npz)
    if train_mask is None:
        train_mask = z["is_train"]

    def w(y):
        y = y[train_mask]
        pos = int(y.sum())
        return float((len(y) - pos) / pos) if pos else 0.0

    return {"detection": w(z["y_progress"]), "warning": w(z["y_warn"])}
