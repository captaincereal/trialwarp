"""Helpers for spike-train data: binning spikes into trial arrays and smoothing them."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter1d

from ._utils import as_index_array, as_trials_array

__all__ = ["bin_spikes", "smooth_trials"]


def bin_spikes(
    trials: ArrayLike,
    times: ArrayLike,
    units: ArrayLike,
    *,
    n_trials: int,
    n_units: int,
    n_bins: int,
    tmin: float = 0.0,
    tmax: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Count spikes in equal time bins for every trial and unit.

    Parameters
    ----------
    trials, times, units : array_like, shape (n_spikes,)
        Trial index, time within the trial, and unit (neuron) index of every
        spike. Spikes outside ``[tmin, tmax]`` are dropped.
    n_trials, n_units, n_bins : int
        Output size.
    tmin, tmax : float
        Window to bin, in the units of ``times``.

    Returns
    -------
    counts : ndarray, shape (n_trials, n_bins, n_units)
        Spike counts as floats.
    centers : ndarray, shape (n_bins,)
        Bin centers. A model fitted on ``counts`` treats ``centers[0]`` and
        ``centers[-1]`` as normalized times 0 and 1, so pass them as ``tmin``
        and ``tmax`` to :meth:`~trialwarp.PiecewiseWarping.transform_events`.
    """
    trials = as_index_array(trials, "trials").ravel()
    units = as_index_array(units, "units").ravel()
    times = np.asarray(times, dtype=float).ravel()
    if not trials.size == times.size == units.size:
        raise ValueError("trials, times and units must have the same length")
    if not tmax > tmin:
        raise ValueError("tmax must be greater than tmin")
    keep = (times >= tmin) & (times <= tmax)
    trials, times, units = trials[keep], times[keep], units[keep]
    if trials.size and (trials.min() < 0 or trials.max() >= n_trials):
        raise IndexError(f"trial indices must be in [0, {n_trials})")
    if units.size and (units.min() < 0 or units.max() >= n_units):
        raise IndexError(f"unit indices must be in [0, {n_units})")
    b = np.minimum(((times - tmin) / (tmax - tmin) * n_bins).astype(np.intp), n_bins - 1)
    flat = (trials * n_bins + b) * n_units + units
    counts = np.bincount(flat, minlength=n_trials * n_bins * n_units).astype(float)
    width = (tmax - tmin) / n_bins
    centers = tmin + width * (np.arange(n_bins) + 0.5)
    return counts.reshape(n_trials, n_bins, n_units), centers


def smooth_trials(X: ArrayLike, sigma: float) -> np.ndarray:
    """Gaussian smoothing along the time axis of every trial and feature.

    Missing (NaN) samples are skipped by normalized convolution and stay NaN,
    so NaN-padded trials of different lengths keep their length.

    Parameters
    ----------
    X : array_like, shape (n_trials, n_timepoints, n_features) or (n_trials, n_timepoints)
    sigma : float
        Standard deviation of the Gaussian kernel, in samples.
    """
    X3, was_2d = as_trials_array(X)
    finite = np.isfinite(X3)
    if finite.all():
        out = gaussian_filter1d(X3, sigma, axis=1, mode="nearest")
    else:
        num = gaussian_filter1d(np.where(finite, X3, 0.0), sigma, axis=1, mode="nearest")
        den = gaussian_filter1d(finite.astype(float), sigma, axis=1, mode="nearest")
        out = np.full_like(num, np.nan)
        np.divide(num, den, out=out, where=finite & (den > 1e-12))
    return out[..., 0] if was_2d else out
