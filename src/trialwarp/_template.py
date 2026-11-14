"""Template interpolation and the banded least-squares template update."""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse


def interp_index(positions: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Left template bin and right-hand interpolation weight for template positions.

    Positions are in normalized template time and are clamped to [0, 1], so a
    warp that runs past either end of the template reads the edge value.
    """
    f = np.clip(positions, 0.0, 1.0) * (n_bins - 1)
    i0 = np.minimum(f.astype(np.intp), n_bins - 2)
    return i0, f - i0


def warp_template(
    template: np.ndarray, template_diff: np.ndarray, i0: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Read ``template`` (n_bins, n_features) at fractional bins; returns (..., n_features).

    ``template_diff`` is ``np.diff(template, axis=0)``, passed in so repeated calls
    do not recompute it.
    """
    out = template_diff[i0]
    out *= w[..., np.newaxis]
    out += template[i0]
    return out


def roughness_bands(n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonals 0, 1 and 2 of ``D.T @ D`` for the second-difference operator ``D``."""
    d0 = np.zeros(n_bins)
    d1 = np.zeros(n_bins - 1)
    d2 = np.ones(n_bins - 2)
    # each row of D is [1, -2, 1] starting at column r, for r = 0 .. n_bins - 3
    d0[:-2] += 1.0
    d0[1:-1] += 4.0
    d0[2:] += 1.0
    d1[:-1] -= 2.0
    d1[1:] -= 2.0
    return d0, d1, d2


def fit_template(
    X: np.ndarray,
    mask: np.ndarray | None,
    i0: np.ndarray,
    w: np.ndarray,
    n_bins: int,
    *,
    roughness: float,
    ridge: float,
) -> np.ndarray:
    """Least-squares template for fixed warps.

    Minimizes::

        sum(mask * (X - W @ T)**2) + c * (roughness * ||D T||^2 + ridge * ||T||^2)

    where ``W`` linearly interpolates the template at each trial sample, ``D``
    takes second differences along template time, and ``c`` is the average
    number of observations per template bin and feature. Scaling the penalties
    by ``c`` keeps their effect independent of how many trials there are.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_timepoints, n_features)
        Data, with missing entries set to zero.
    mask : None or ndarray of 0/1, shape (n_trials, n_timepoints) or X.shape
        Observed entries; ``None`` means all are observed. A 2D mask applies to
        every feature, which lets all features share one linear system.
    i0, w : ndarray, shape (n_trials, n_timepoints)
        Left template bin and interpolation weight of every sample (see
        :func:`interp_index`).

    Returns
    -------
    ndarray, shape (n_bins, n_features)
    """
    n_trials, n_time, n_features = X.shape
    rows = i0.ravel()
    wr = w.ravel()
    wl = 1.0 - wr
    flat = X.reshape(n_trials * n_time, n_features)
    rough = roughness_bands(n_bins)

    def system(diag: np.ndarray, off: np.ndarray, scale: float) -> np.ndarray:
        # symmetric pentadiagonal matrix in the upper banded form of solveh_banded
        ab = np.zeros((3, n_bins))
        ab[2] = diag + scale * (roughness * rough[0] + ridge)
        ab[2] += 1e-10 * max(float(ab[2].max()), 1.0)  # keeps empty bins solvable
        ab[1, 1:] = off + scale * roughness * rough[1]
        ab[0, 2:] = scale * roughness * rough[2]
        return ab

    if mask is None or mask.ndim == 2:
        m = np.ones(rows.size) if mask is None else mask.ravel()
        diag = np.bincount(rows, wl * wl * m, n_bins) + np.bincount(rows + 1, wr * wr * m, n_bins)
        off = np.bincount(rows, wl * wr * m, n_bins)[: n_bins - 1]
        cols = np.arange(rows.size)
        interp = sparse.csr_array(
            (
                np.concatenate([wl * m, wr * m]),
                (np.concatenate([rows, rows + 1]), np.concatenate([cols, cols])),
            ),
            shape=(n_bins, rows.size),
        )
        rhs = interp @ flat
        return _solve(system(diag, off, m.sum() / n_bins), rhs)

    # entries missing feature by feature: one system per feature, shared penalty scale
    scale = mask.sum() / (n_bins * n_features)
    out = np.empty((n_bins, n_features))
    for n in range(n_features):
        m = mask[..., n].ravel()
        diag = np.bincount(rows, wl * wl * m, n_bins) + np.bincount(rows + 1, wr * wr * m, n_bins)
        off = np.bincount(rows, wl * wr * m, n_bins)[: n_bins - 1]
        xm = flat[:, n] * m
        rhs = np.bincount(rows, wl * xm, n_bins) + np.bincount(rows + 1, wr * xm, n_bins)
        out[:, n] = _solve(system(diag, off, scale), rhs)
    return out


def _solve(ab: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return linalg.solveh_banded(ab, rhs, lower=False, check_finite=False)
    except linalg.LinAlgError:
        # not numerically positive definite: fall back to a general banded LU solve
        full = np.zeros((5, ab.shape[1]))
        full[:3] = ab
        full[3, :-1] = ab[1, 1:]
        full[4, :-2] = ab[0, 2:]
        return linalg.solve_banded((2, 2), full, rhs, check_finite=False)
