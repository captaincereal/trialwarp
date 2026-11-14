"""Small internal helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def as_trials_array(X: ArrayLike, name: str = "X") -> tuple[np.ndarray, bool]:
    """Return ``X`` as a float array of shape (n_trials, n_timepoints, n_features).

    A 2D input of shape (n_trials, n_timepoints) is treated as one feature. The
    second return value reports this so callers can drop the axis again.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 2:
        return arr[:, :, np.newaxis], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError(
        f"{name} must have shape (n_trials, n_timepoints, n_features) or "
        f"(n_trials, n_timepoints); got an array with shape {arr.shape}"
    )


def check_random_state(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Turn ``None``, an int seed or a Generator into a Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def as_index_array(values: ArrayLike, name: str) -> np.ndarray:
    """Integer index array; rejects non-integral floats instead of truncating them."""
    arr = np.asarray(values)
    if arr.dtype.kind in "iu":
        return arr.astype(np.intp, copy=False)
    as_int = arr.astype(np.intp)
    if not np.array_equal(as_int, arr):
        raise TypeError(f"{name} must contain integer indices")
    return as_int
