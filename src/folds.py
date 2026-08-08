# walk-forward (rolling-origin) fold masks over event chronology. each fold
# trains on the past, validates on the next block, tests on the block after -
# so every test event was predicted by a model that only saw its past. pooling
# the test blocks gives far more than 117/29 events, which is the whole point:
# tighter CIs. the load-bearing invariant: no flare's windows straddle a split,
# or its tail would leak into training.

import numpy as np


def _block_of_window(t_end, n_blocks):
    # contiguous blocks by RANK of t_end, not raw time, so blocks hold ~equal
    # window counts even when solar activity is bursty.
    order = np.argsort(t_end, kind="stable")
    rank = np.empty(len(t_end), dtype=np.int64)
    rank[order] = np.arange(len(t_end))
    return (rank * n_blocks // len(t_end)).astype(np.int64)


def block_labels(t_end, event_id, n_blocks):
    # window -> block, then pull every event's windows into one block (majority)
    # so no flare spans a boundary. quiet windows (event_id<0) stay time-based.
    blk = _block_of_window(t_end, n_blocks)
    for e in np.unique(event_id[event_id >= 0]):
        sel = event_id == e
        maj = np.bincount(blk[sel]).argmax()
        blk[sel] = maj
    return blk


def walk_forward_folds(t_end, event_id, n_blocks=6):
    # fold f: train blocks [0..f-1], val block f, test block f+1, for f=1..n-2.
    # returns list of (train_mask, val_mask, test_mask). test blocks are disjoint
    # across folds, so their events pool cleanly.
    blk = block_labels(t_end, event_id, n_blocks)
    folds = []
    for f in range(1, n_blocks - 1):
        folds.append((blk < f, blk == f, blk == f + 1))
    return folds
