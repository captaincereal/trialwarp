"""Checking whether warping helps: feature hold-out validation and trial consistency.

Warping always lowers the training loss, even on pure noise, so a lower loss
says little. What matters is whether warps learned from some features also
line up features they never saw. :func:`heldout_feature_score` measures that
with a double cross-validation: warps come from the other features, the
template from the other trials, and the held-out data are compared in their
raw form, never resampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._template import fit_template, interp_index, warp_template
from ._utils import as_trials_array, check_random_state

__all__ = ["HeldoutScore", "clone", "heldout_feature_score", "trial_consistency"]


def clone(model):
    """An unfitted copy of ``model`` with the same parameters."""
    return type(model)(**model.get_params())


def trial_consistency(X: ArrayLike, mask: ArrayLike | None = None) -> float:
    """Leave-one-out R²: how well the average of the other trials predicts each trial.

    For every trial, the prediction is the mean of all other trials at the same
    timepoint and feature. The score is ``1 - SS_residual / SS_total`` with
    ``SS_total`` taken around each feature's overall mean, so 1 means all
    trials are identical and values near 0 mean they share no time course.
    NaN entries (and entries outside ``mask``) are ignored.

    Use it to compare alignments of noise-free data, such as the ``clean``
    signals of a synthetic data set. On noisy data, alignment resamples trials
    by interpolation, which averages neighboring samples, damps noise, and
    raises this score a little by itself; use :func:`heldout_feature_score` to
    decide whether warping helps on real data.

    Parameters
    ----------
    X : array_like, shape (n_trials, n_timepoints, n_features) or (n_trials, n_timepoints)
    mask : array_like of bool, optional
        Entries to include, same shape as ``X``.
    """
    X3, was_2d = as_trials_array(X)
    valid = np.isfinite(X3)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        valid &= m[..., np.newaxis] if was_2d else m
    Y = np.where(valid, X3, 0.0)
    total = Y.sum(axis=0)
    others = valid.sum(axis=0) - valid  # number of other trials observed at each entry
    use = valid & (others > 0)
    loo_mean = (total - Y) / np.maximum(others, 1)
    resid = np.where(use, Y - loo_mean, 0.0)
    feature_mean = Y.sum(axis=(0, 1)) / np.maximum(valid.sum(axis=(0, 1)), 1)
    dev = np.where(use, Y - feature_mean, 0.0)
    ss_total = float(np.sum(dev**2))
    if ss_total == 0.0:
        return float("nan")
    return 1.0 - float(np.sum(resid**2)) / ss_total


@dataclass(frozen=True)
class HeldoutScore:
    """Result of :func:`heldout_feature_score`, one entry per fold of features.

    Attributes
    ----------
    raw : ndarray
        Cross-validated R² for the held-out features without warping (the
        template is then a smoothed trial average).
    aligned : ndarray
        The same R² when trials are read through warps fitted on the other
        features.
    """

    raw: np.ndarray
    aligned: np.ndarray

    @property
    def improvement(self) -> np.ndarray:
        """``aligned - raw`` per fold; consistently positive means the warps generalize."""
        return self.aligned - self.raw

    def __str__(self) -> str:
        def fmt(a: np.ndarray) -> str:
            return f"{a.mean():.3f} +/- {a.std():.3f}"

        return (
            f"held-out R^2  unwarped {fmt(self.raw)}  warped {fmt(self.aligned)}  "
            f"improvement {fmt(self.improvement)}  ({self.raw.size} folds)"
        )


def heldout_feature_score(
    model,
    X: ArrayLike,
    *,
    n_folds: int = 5,
    n_trial_folds: int = 5,
    random_state: int | np.random.Generator | None = None,
    **fit_kwargs: Any,
) -> HeldoutScore:
    """Cross-validate a warping model on features it was not fitted on.

    The features are split at random into ``n_folds`` groups. For each group:

    1. a fresh copy of ``model`` is fitted on all *other* features, giving
       one warp per trial;
    2. the held-out features are scored by cross-validation over trials: a
       template is fitted on all but one fold of trials and must predict the
       remaining trials, read out through their warps;
    3. the same is done with identity warps, as the unwarped baseline.

    The held-out data are never resampled, and both predictions use the same
    template fitting (``smoothness``, ``l2_reg``, ``n_bins`` of ``model``), so
    the comparison isolates the value of the warps.

    Parameters
    ----------
    model : trialwarp model
        Only its parameters are used; it is not modified.
    X : array_like, shape (n_trials, n_timepoints, n_features)
        Needs at least two features and two trials. NaN marks missing data.
    n_folds : int, default 5
        Number of feature groups (capped at the number of features).
    n_trial_folds : int, default 5
        Number of trial groups used to score each feature group.
    random_state : int, Generator or None
        Seed for splitting features and trials into folds.
    **fit_kwargs
        Passed to ``fit`` (for example ``n_iter=10``).

    Returns
    -------
    HeldoutScore
        Per feature group, R² without and with warping.
    """
    X3, _ = as_trials_array(X)
    n_trials, n_time, n_features = X3.shape
    if n_features < 2:
        raise ValueError("feature hold-out needs at least two features")
    if n_trials < 2:
        raise ValueError("feature hold-out needs at least two trials")
    if n_folds < 2 or n_trial_folds < 2:
        raise ValueError("n_folds and n_trial_folds must be at least 2")
    rng = check_random_state(random_state)
    feature_order = rng.permutation(n_features)
    trial_folds = np.array_split(rng.permutation(n_trials), min(n_trial_folds, n_trials))
    t = np.linspace(0.0, 1.0, n_time)

    raw, aligned = [], []
    for test in np.array_split(feature_order, min(n_folds, n_features)):
        train = np.setdiff1d(feature_order, test)
        fitted = clone(model).fit(X3[:, :, train], **fit_kwargs)
        held = X3[:, :, test]
        warps = fitted.warping_functions(t)
        raw.append(_crossval_r2(held, np.broadcast_to(t, warps.shape), trial_folds, model))
        aligned.append(_crossval_r2(held, warps, trial_folds, model))
    return HeldoutScore(raw=np.asarray(raw), aligned=np.asarray(aligned))


def _crossval_r2(Y: np.ndarray, warps: np.ndarray, trial_folds: list[np.ndarray], model) -> float:
    """R² of predicting each fold of trials from a template fitted on the other folds."""
    n_trials, n_time, _ = Y.shape
    n_bins = n_time if model.n_bins is None else int(model.n_bins)
    roughness = (model.smoothness * (n_bins - 1)) ** 4
    i0, w = interp_index(warps, n_bins)
    finite = np.isfinite(Y)
    sse = sst = 0.0
    for test in trial_folds:
        train = np.setdiff1d(np.arange(n_trials), test)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nansum(Y[train], axis=(0, 1)) / finite[train].sum(axis=(0, 1))
        mean = np.where(np.isfinite(mean), mean, 0.0)
        centered = np.where(finite, Y - mean, 0.0)
        mask = None if finite[train].all() else finite[train].astype(float)
        template = fit_template(
            centered[train],
            mask,
            i0[train],
            w[train],
            n_bins,
            roughness=roughness,
            ridge=model.l2_reg,
        )
        pred = warp_template(template, np.diff(template, axis=0), i0[test], w[test])
        observed = finite[test]
        sse += float(np.sum((centered[test] - pred)[observed] ** 2))
        sst += float(np.sum(centered[test][observed] ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")
