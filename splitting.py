"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Sweep knob. Valid values: "single" (default), "5fold".
SPLIT_STRATEGY = os.environ.get("SPLIT_STRATEGY", "single")
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "42"))


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = SPLIT_SEED,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs a single stratified random split preserving
    the class ratio in each subset.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    idx = np.arange(len(y))

    if SPLIT_STRATEGY == "single":
        idx_train_val, idx_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        relative_val = val_size / (1.0 - test_size)
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=relative_val,
            random_state=random_state,
            stratify=y[idx_train_val],
        )
        return [(idx_train, idx_val, idx_test)]

    if SPLIT_STRATEGY == "5fold":
        # Stratified 5-fold; carve a val slice from each fold's train for
        # threshold tuning via probe.fit_hyperparameters().
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
        for tr_idx, te_idx in skf.split(idx, y):
            idx_tr, idx_va = train_test_split(
                tr_idx,
                test_size=val_size / (1.0 - 1.0 / 5),
                random_state=random_state,
                stratify=y[tr_idx],
            )
            splits.append((idx_tr, idx_va, te_idx))
        return splits

    raise ValueError(f"unknown SPLIT_STRATEGY={SPLIT_STRATEGY!r}")

