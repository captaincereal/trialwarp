"""Piecewise-linear warping functions.

A warping function is stored as knots ``(x, y)``: it passes through
``(x[j], y[j])``, is linear between knots, and continues linearly past the
first and last knot. Batches of warps are arrays of shape
``(n_warps, n_points)``, with ``x`` strictly increasing along the last axis.

In trialwarp models ``x`` is normalized trial time and ``y`` is normalized
template time, so ``x = [0, 1], y = [0.1, 1.1]`` is a warp under which the
trial runs 10% of a trial length ahead of the template.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def evaluate_warps(x_knots: ArrayLike, y_knots: ArrayLike, t: ArrayLike) -> np.ndarray:
    """Evaluate a batch of piecewise-linear warping functions.

    Parameters
    ----------
    x_knots, y_knots : array_like, shape (n_warps, n_points)
        Knot coordinates. ``x_knots`` must be strictly increasing along axis 1.
    t : array_like, shape (n_times,) or (n_warps, n_times)
        Points to evaluate, either shared by all warps or one row per warp.

    Returns
    -------
    ndarray, shape (n_warps, n_times)
        ``out[k, i]`` is warp ``k`` evaluated at ``t[i]`` (or ``t[k, i]``).
    """
    x = np.asarray(x_knots, dtype=float)
    y = np.asarray(y_knots, dtype=float)
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] < 2:
        raise ValueError(
            "x_knots and y_knots must have the same shape (n_warps, n_points) with n_points >= 2"
        )
    t = np.asarray(t, dtype=float)
    if t.ndim == 1:
        t = np.broadcast_to(t, (x.shape[0], t.size))
    elif t.ndim != 2 or t.shape[0] != x.shape[0]:
        raise ValueError("t must have shape (n_times,) or (n_warps, n_times)")

    n_warps, n_points = x.shape
    # flat index of the knot that starts the segment holding each point
    idx = np.arange(n_warps)[:, np.newaxis] * n_points
    if n_points > 2:
        idx = idx + np.count_nonzero(t[:, :, np.newaxis] >= x[:, np.newaxis, 1:-1], axis=2)
    else:
        idx = np.broadcast_to(idx, t.shape)
    xf, yf = x.ravel(), y.ravel()
    x0, x1 = xf[idx], xf[idx + 1]
    y0, y1 = yf[idx], yf[idx + 1]
    return y0 + (y1 - y0) * ((t - x0) / (x1 - x0))


def invert_warps(x_knots: ArrayLike, y_knots: ArrayLike, s: ArrayLike) -> np.ndarray:
    """Evaluate the inverse warping functions (template time back to trial time).

    The warps must be strictly increasing, which every trialwarp model guarantees.
    Arguments and output are as for :func:`evaluate_warps`.
    """
    return evaluate_warps(y_knots, x_knots, s)


def warp_penalty(x_knots: ArrayLike, y_knots: ArrayLike) -> np.ndarray:
    """Exact integral of the squared displacement, ``∫_0^1 (ω(t) - t)^2 dt``, per warp.

    Assumes the first and last knots sit at ``x = 0`` and ``x = 1``.
    """
    x = np.asarray(x_knots, dtype=float)
    d = np.asarray(y_knots, dtype=float) - x
    length = np.diff(x, axis=-1)
    a, b = d[..., :-1], d[..., 1:]
    # integral of a linear function's square over each segment
    return np.sum(length * (a * a + a * b + b * b), axis=-1) / 3.0


def identity_knots(n_warps: int, n_knots: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Knots of ``n_warps`` identity warps with ``n_knots`` evenly spaced interior knots."""
    x = np.tile(np.linspace(0.0, 1.0, n_knots + 2), (n_warps, 1))
    return x, x.copy()


def random_knots(
    n_warps: int,
    n_knots: int,
    *,
    max_lag: float,
    max_stretch: float,
    rng: np.random.Generator,
    log_slope_sd: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw random increasing piecewise-linear warps with evenly spaced knots.

    Each segment's slope is ``exp`` of a shared plus an independent normal draw
    (clipped to ``[1/max_stretch, max_stretch]``). The warp is then shifted by an
    offset drawn uniformly from the range that keeps ``|ω(t) - t| <= max_lag``.
    Draws whose shape cannot satisfy the lag bound are discarded, so in rare,
    tightly constrained cases fewer than ``n_warps`` warps are returned.
    """
    n_points = n_knots + 2
    grid = np.linspace(0.0, 1.0, n_points)
    limit = np.log(max_stretch)
    found: list[np.ndarray] = []
    n_found = 0
    for _ in range(20):
        if n_found >= n_warps:
            break
        m = 2 * (n_warps - n_found) + 8
        log_slope = rng.normal(0.0, log_slope_sd, (m, 1)) + rng.normal(
            0.0, log_slope_sd, (m, n_points - 1)
        )
        slope = np.exp(np.clip(log_slope, -limit, limit))
        y = np.zeros((m, n_points))
        y[:, 1:] = np.cumsum(slope * np.diff(grid), axis=1)
        d = y - grid
        lo = -max_lag - d.min(axis=1)
        hi = max_lag - d.max(axis=1)
        ok = lo <= hi
        y = y[ok] + rng.uniform(lo[ok], hi[ok])[:, np.newaxis]
        found.append(y)
        n_found += len(y)
    y = np.concatenate(found)[:n_warps] if found else np.empty((0, n_points))
    return np.tile(grid, (len(y), 1)), y
