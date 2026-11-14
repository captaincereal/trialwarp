"""Time-warping models for aligning repeated trials.

Every model describes trial ``k`` with an increasing, piecewise-linear warping
function ``ω_k`` that maps normalized trial time ``t`` in [0, 1] onto
normalized template time. The prediction for trial ``k`` is the shared
template read out at ``ω_k(t)``::

    X[k, i, :]  ≈  template(ω_k(t_i)),      t_i = i / (n_timepoints - 1)

Fitting alternates two steps, each of which lowers the same objective:

1. **Warps.** Every trial is scored against a shared dictionary of candidate
   warps (a coarse shift/stretch grid plus random piecewise warps) with a few
   matrix products, and switches to the best candidate if it beats its current
   warp. A short stochastic local search then refines each trial's warp.
2. **Template.** With the warps fixed the prediction is linear in the
   template, so the best template solves a banded least-squares problem with a
   roughness penalty.

The objective is normalized so that the regularization weights mean the same
thing on any data set::

    loss = mean squared residual / variance of X
           + warp_reg * mean over trials of ∫_0^1 (ω_k(t) - t)^2 dt
           + template roughness penalty
"""

from __future__ import annotations

import inspect
import math
import time
import warnings
from typing import Any, NamedTuple, Self

import numpy as np
from numpy.typing import ArrayLike

from ._template import fit_template, interp_index, warp_template
from ._utils import as_index_array, as_trials_array, check_random_state
from .warps import evaluate_warps, identity_knots, invert_warps, random_knots, warp_penalty

__all__ = ["LinearWarping", "PiecewiseWarping", "ShiftWarping"]

# Largest temporary array (in float64 elements) built while scoring warps.
_CHUNK_ELEMENTS = 2_000_000
# Per-trial step sizes of the local search follow the 1/5th success rule.
_GROW = math.exp(1.0 / 3.0)
_SHRINK = math.exp(-1.0 / 12.0)


class _Space(NamedTuple):
    """What warp scores are computed from: data (zero where missing), its mask, the
    template in the same feature basis, and every trial's sum of squares."""

    X: np.ndarray
    mask: np.ndarray | None
    template: np.ndarray
    template_diff: np.ndarray
    xnorm: np.ndarray


class _BaseWarping:
    """Fitting and transform machinery shared by the public models."""

    def __init__(
        self,
        *,
        max_lag: float,
        warp_reg: float,
        smoothness: float,
        l2_reg: float,
        n_bins: int | None,
    ) -> None:
        self.max_lag = max_lag
        self.warp_reg = warp_reg
        self.smoothness = smoothness
        self.l2_reg = l2_reg
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    # parameters

    @classmethod
    def _param_names(cls) -> list[str]:
        params = inspect.signature(cls.__init__).parameters.values()
        return [p.name for p in params if p.name != "self"]

    def get_params(self) -> dict[str, Any]:
        """Return the constructor arguments of this model as a dict."""
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> Self:
        """Set constructor arguments by keyword; returns the model."""
        names = set(self._param_names())
        for key, value in params.items():
            if key not in names:
                raise ValueError(f"{key!r} is not a parameter of {type(self).__name__}")
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    def _validate_params(self) -> None:
        if not 0.0 < self.max_lag <= 1.0:
            raise ValueError(f"max_lag must be in (0, 1], got {self.max_lag}")
        for name in ("warp_reg", "smoothness", "l2_reg"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.n_bins is not None and int(self.n_bins) < 3:
            raise ValueError(f"n_bins must be at least 3, got {self.n_bins}")

    # ------------------------------------------------------------------
    # hooks for the warp family (overridden by subclasses)

    @property
    def _n_points(self) -> int:
        """Number of knots per warp, endpoints included."""
        raise NotImplementedError

    def _slope_bounds(self) -> tuple[float, float]:
        raise NotImplementedError

    def _setup_search(self) -> None:
        """Precompute the candidate grid and step sizes once the data size is known."""
        raise NotImplementedError

    def _candidate_knots(
        self, rng: np.random.Generator | None, n_candidates: int
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _propose(
        self, x: np.ndarray, y: np.ndarray, step: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _project(self, t: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Knots of this family that best follow warps sampled as ``values`` at ``t``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # fitting

    def _fit(
        self,
        X: ArrayLike,
        *,
        n_iter: int,
        init: _BaseWarping | None,
        verbose: bool,
        rng: np.random.Generator | None,
        n_proposals: int,
        n_candidates: int,
    ) -> Self:
        self._validate_params()
        if n_iter < 0:
            raise ValueError(f"n_iter must be non-negative, got {n_iter}")
        X3, _ = as_trials_array(X)
        n_trials, n_time, _ = X3.shape
        if n_trials < 2:
            raise ValueError("need at least two trials to learn a shared template")
        if n_time < 3:
            raise ValueError("need at least three timepoints per trial")

        self._prepare(X3)
        if init is None:
            self._x, self._y = identity_knots(n_trials, self._n_points - 2)
        else:
            self._x, self._y = self._knots_from_model(init)
        self._update_template()
        history = [self._loss()]
        start = time.perf_counter()
        for it in range(n_iter):
            self._update_warps(rng, n_proposals, n_candidates)
            self._update_template()
            history.append(self._loss())
            if verbose:
                elapsed = time.perf_counter() - start
                print(
                    f"iter {it + 1:>4}/{n_iter}   loss {history[-1]:.6f}   {elapsed:6.1f}s",
                    flush=True,
                )
        self._finish(history)
        return self

    def _prepare(self, X3: np.ndarray) -> None:
        n_trials, n_time, n_features = X3.shape
        finite = np.isfinite(X3)
        n_obs = int(finite.sum())
        if n_obs == 0:
            raise ValueError("X contains no finite values")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-missing features
            mean = np.nanmean(X3, axis=(0, 1))
        mean = np.where(np.isfinite(mean), mean, 0.0)
        Xc = np.where(finite, X3 - mean, 0.0)
        if finite.all():
            mask = None
        elif (finite == finite[..., :1]).all():
            mask = finite[..., 0].astype(float)  # whole samples missing, e.g. ragged trials
        else:
            mask = finite.astype(float)
        variance = float(np.einsum("ktn,ktn->", Xc, Xc)) / n_obs
        if not variance > 0:
            variance = 1.0

        self._X = Xc
        self._mask = mask
        self._feature_mean = mean
        self._variance = variance
        self._alpha = n_trials / (n_obs * variance)
        self._xnorm = np.einsum("ktn,ktn->k", Xc, Xc)
        self._t = np.linspace(0.0, 1.0, n_time)
        self._n_template = n_time if self.n_bins is None else int(self.n_bins)
        # a roughness weight of (s * n_bins)^4 smooths over roughly s of the trial
        self._roughness = (self.smoothness * (self._n_template - 1)) ** 4
        self._chunk = max(1, _CHUNK_ELEMENTS // (n_time * n_features))
        self._min_gap = min(1.0 / (n_time - 1), 0.5 / (self._n_points - 1))
        self._setup_search()

    def _finish(self, history: list[float]) -> None:
        self.x_knots_ = self._x.copy()
        self.y_knots_ = self._y.copy()
        self.template_ = self._template + self._feature_mean
        self.loss_history_ = np.asarray(history)
        self.n_trials_, self.n_timepoints_, self.n_features_ = self._X.shape
        for name in ("_X", "_mask", "_xnorm", "_x", "_y", "_template", "_template_diff"):
            delattr(self, name)  # drop fitting buffers, including the copy of the data

    def _knots_from_model(self, init: _BaseWarping) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(init, "x_knots_"):
            raise TypeError("init must be a fitted trialwarp model")
        n_trials = self._X.shape[0]
        if init.x_knots_.shape[0] != n_trials:
            raise ValueError(
                f"init was fitted on {init.x_knots_.shape[0]} trials, but X has {n_trials}"
            )
        t = np.linspace(0.0, 1.0, max(self._t.size, 101))
        x, y = self._project(t, init.warping_functions(t))
        if init.x_knots_.shape[1] == self._n_points:
            # same number of knots: keep the exact knots wherever they satisfy our constraints
            same = self._is_valid(init.x_knots_, init.y_knots_)
            x[same] = init.x_knots_[same]
            y[same] = init.y_knots_[same]
        return x, y

    def _update_template(self) -> None:
        warp = evaluate_warps(self._x, self._y, self._t)
        i0, w = interp_index(warp, self._n_template)
        self._template = fit_template(
            self._X,
            self._mask,
            i0,
            w,
            self._n_template,
            roughness=self._roughness,
            ridge=self.l2_reg,
        )
        self._template_diff = np.diff(self._template, axis=0)

    def _loss(self) -> float:
        per_trial = self._trial_objective(self._full_space(), self._x, self._y)
        tmpl = self._template
        penalty = self._roughness * np.sum(np.diff(tmpl, n=2, axis=0) ** 2)
        penalty += self.l2_reg * np.sum(tmpl**2)
        return float(per_trial.mean() + penalty / (tmpl.size * self._variance))

    def _full_space(self) -> _Space:
        return _Space(self._X, self._mask, self._template, self._template_diff, self._xnorm)

    def _search_space(self) -> _Space:
        """Space for scoring warps: the template's principal axes, when much smaller.

        Predictions always lie in the span of the template's rows, so the part
        of the data orthogonal to that span adds the same amount to every warp's
        residual. Dropping it changes each trial's score by a constant and
        leaves comparisons between warps intact (up to the 1e-10 share of
        template energy that is truncated), while scoring gets cheaper by the
        ratio of features to template rank.
        """
        n_trials, n_time, n_features = self._X.shape
        if self._mask is not None or n_features < 8:
            return self._full_space()
        u, s, vt = np.linalg.svd(self._template, full_matrices=False)
        energy = np.cumsum(s**2)
        if not energy[-1] > 0:
            return self._full_space()
        rank = int(np.searchsorted(energy, energy[-1] * (1.0 - 1e-10))) + 1
        if rank > 0.7 * n_features:
            return self._full_space()
        X = (self._X.reshape(-1, n_features) @ vt[:rank].T).reshape(n_trials, n_time, rank)
        template = u[:, :rank] * s[:rank]
        return _Space(X, None, template, np.diff(template, axis=0), np.einsum("ktr,ktr->k", X, X))

    def _trial_sse(
        self, space: _Space, warp: np.ndarray, rows: np.ndarray | None = None
    ) -> np.ndarray:
        """Masked sum of squared residuals per trial for template positions ``warp``.

        ``warp`` has one row per trial in ``rows``, or per trial when ``rows`` is None.
        """
        n = len(warp)
        out = np.empty(n)
        for start in range(0, n, self._chunk):
            stop = min(start + self._chunk, n)
            sel = slice(start, stop) if rows is None else rows[start:stop]
            i0, w = interp_index(warp[start:stop], self._n_template)
            resid = warp_template(space.template, space.template_diff, i0, w)
            resid -= space.X[sel]
            if space.mask is not None:
                m = space.mask[sel]
                resid *= m if m.ndim == 3 else m[..., np.newaxis]
            out[start:stop] = np.einsum("ktn,ktn->k", resid, resid)
        return out

    def _trial_objective(
        self, space: _Space, x: np.ndarray, y: np.ndarray, rows: np.ndarray | None = None
    ) -> np.ndarray:
        warp = evaluate_warps(x, y, self._t)
        sse = self._trial_sse(space, warp, rows)
        return self._alpha * sse + self.warp_reg * warp_penalty(x, y)

    def _search_candidates(
        self, space: _Space, xc: np.ndarray, yc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score every trial against every candidate warp; return each trial's best.

        Candidates are shared by all trials, so with ``P_g`` the template warped
        by candidate ``g`` the residuals expand into matrix products::

            ||X_k - P_g||^2 = ||X_k||^2 - 2 <X_k, P_g> + ||P_g||^2
        """
        n_trials, n_time, n_dims = space.X.shape
        flat = space.X.reshape(n_trials, n_time * n_dims)
        warp = evaluate_warps(xc, yc, self._t)
        penalty = self.warp_reg * warp_penalty(xc, yc)
        best_val = np.full(n_trials, np.inf)
        best_idx = np.zeros(n_trials, dtype=np.intp)
        rows = np.arange(n_trials)
        step = max(1, _CHUNK_ELEMENTS // (n_time * n_dims))
        for start in range(0, len(xc), step):
            stop = min(start + step, len(xc))
            i0, w = interp_index(warp[start:stop], self._n_template)
            pred = warp_template(space.template, space.template_diff, i0, w)
            P = pred.reshape(stop - start, n_time * n_dims)
            if space.mask is None:
                pnorm = np.einsum("gj,gj->g", P, P)[np.newaxis, :]
            elif space.mask.ndim == 2:
                pnorm = space.mask @ np.einsum("gtn,gtn->gt", pred, pred).T
            else:
                pnorm = space.mask.reshape(n_trials, -1) @ (P * P).T
            sse = space.xnorm[:, np.newaxis] - 2.0 * (flat @ P.T) + pnorm
            obj = self._alpha * sse + penalty[np.newaxis, start:stop]
            j = np.argmin(obj, axis=1)
            val = obj[rows, j]
            better = val < best_val
            best_val[better] = val[better]
            best_idx[better] = j[better] + start
        return best_idx, best_val

    def _update_warps(
        self, rng: np.random.Generator | None, n_proposals: int, n_candidates: int
    ) -> None:
        space = self._search_space()
        current = self._trial_objective(space, self._x, self._y)

        xc, yc = self._candidate_knots(rng, n_candidates)
        if len(xc):
            best, val = self._search_candidates(space, xc, yc)
            better = val < current
            self._x[better] = xc[best[better]]
            self._y[better] = yc[best[better]]
            current[better] = val[better]

        if n_proposals <= 0 or rng is None:
            return
        step = np.full(len(self._x), self._step0)
        for _ in range(n_proposals):
            xp, yp = self._propose(self._x, self._y, step, rng)
            ok = self._is_valid(xp, yp)
            # score invalid proposals at the current warp (safe to evaluate, never accepted)
            xp[~ok] = self._x[~ok]
            yp[~ok] = self._y[~ok]
            val = self._trial_objective(space, xp, yp)
            won = ok & (val < current)
            self._x[won] = xp[won]
            self._y[won] = yp[won]
            current[won] = val[won]
            step *= np.where(won, _GROW, _SHRINK)
            np.clip(step, self._step_min, self._step_max, out=step)

    def _is_valid(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Which warps satisfy the knot-spacing, slope and lag constraints."""
        dx = np.diff(x, axis=1)
        dy = np.diff(y, axis=1)
        lo, hi = self._slope_bounds()
        ok = np.all(dx >= self._min_gap * (1.0 - 1e-9), axis=1)
        ok &= np.all(dy >= lo * dx * (1.0 - 1e-9), axis=1)
        ok &= np.all(dy <= hi * dx * (1.0 + 1e-9), axis=1)
        ok &= np.all(np.abs(y - x) <= self.max_lag * (1.0 + 1e-9) + 1e-12, axis=1)
        return ok

    def _line_knots(self, t: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Least-squares straight lines through warp samples, forced into the constraints."""
        tc = t - t.mean()
        slope = (values - values.mean(axis=1, keepdims=True)) @ tc / (tc @ tc)
        lo, hi = self._slope_bounds()
        lag = self.max_lag
        slope = np.clip(slope, max(lo, 1.0 - 2.0 * lag), min(hi, 1.0 + 2.0 * lag))
        offset = values.mean(axis=1) - slope * t.mean()
        # keep both endpoints within max_lag of the identity
        lower = np.maximum(-self.max_lag, -self.max_lag - (slope - 1.0))
        upper = np.minimum(self.max_lag, self.max_lag - (slope - 1.0))
        offset = np.clip(offset, lower, upper)
        return offset[:, np.newaxis] + slope[:, np.newaxis] * grid

    # ------------------------------------------------------------------
    # using a fitted model

    def _check_fitted(self) -> None:
        if not hasattr(self, "x_knots_"):
            raise RuntimeError(f"this {type(self).__name__} is not fitted yet; call fit() first")

    def warping_functions(self, t: ArrayLike | None = None) -> np.ndarray:
        """Evaluate each trial's warp ``ω_k`` (trial time -> template time).

        Parameters
        ----------
        t : array_like, optional
            Normalized trial times in [0, 1]. Defaults to the sample times of the
            training data.

        Returns
        -------
        ndarray, shape (n_trials, len(t))
        """
        self._check_fitted()
        t = np.linspace(0.0, 1.0, self.n_timepoints_) if t is None else np.atleast_1d(t)
        return evaluate_warps(self.x_knots_, self.y_knots_, t)

    def inverse_warping_functions(self, s: ArrayLike | None = None) -> np.ndarray:
        """Evaluate each trial's inverse warp (template time -> trial time)."""
        self._check_fitted()
        s = np.linspace(0.0, 1.0, self.n_timepoints_) if s is None else np.atleast_1d(s)
        return invert_warps(self.x_knots_, self.y_knots_, s)

    def predict(self, n_timepoints: int | None = None) -> np.ndarray:
        """Model reconstruction of every trial: the template warped by each trial's warp.

        Returns
        -------
        ndarray, shape (n_trials, n_timepoints, n_features)
        """
        self._check_fitted()
        n_time = self.n_timepoints_ if n_timepoints is None else int(n_timepoints)
        warp = self.warping_functions(np.linspace(0.0, 1.0, n_time))
        i0, w = interp_index(warp, self.template_.shape[0])
        return warp_template(self.template_, np.diff(self.template_, axis=0), i0, w)

    def transform(self, X: ArrayLike, *, fill_value: float = np.nan) -> np.ndarray:
        """Align trials by resampling each one on the template's clock.

        ``X`` needs the same trials (and order) as the training data, but may hold
        other features or another sampling rate: warps live in normalized time.
        For example, fit on smoothed firing rates and transform raw ones.

        Parameters
        ----------
        X : array_like, shape (n_trials, n_timepoints, n_features) or (n_trials, n_timepoints)
        fill_value : float, default nan
            Value for aligned samples that fall outside a trial's recorded window.

        Returns
        -------
        ndarray, same shape as ``X``
            ``out[k, i]`` is trial ``k`` at the trial time that its warp maps to
            template time ``i / (n_timepoints - 1)``, linearly interpolated.
        """
        self._check_fitted()
        X3, was_2d = as_trials_array(X)
        n_trials, n_time, _ = X3.shape
        if n_trials != self.n_trials_:
            raise ValueError(f"model was fitted on {self.n_trials_} trials, X has {n_trials}")
        if n_time < 2:
            raise ValueError("need at least two timepoints per trial")
        u = invert_warps(self.x_knots_, self.y_knots_, np.linspace(0.0, 1.0, n_time))
        inside = (u >= -1e-9) & (u <= 1.0 + 1e-9)
        f = np.clip(u, 0.0, 1.0) * (n_time - 1)
        i0 = np.minimum(f.astype(np.intp), n_time - 2)[..., np.newaxis]
        w = f[..., np.newaxis] - i0
        left = np.take_along_axis(X3, i0, axis=1)
        right = np.take_along_axis(X3, i0 + 1, axis=1)
        out = np.where(w > 0.0, left + w * (right - left), left)
        out[~inside] = fill_value
        return out[..., 0] if was_2d else out

    def fit_transform(self, X: ArrayLike, **fit_kwargs: Any) -> np.ndarray:
        """Fit the model on ``X`` and return the aligned ``X``."""
        return self.fit(X, **fit_kwargs).transform(X)

    def transform_events(
        self, trials: ArrayLike, times: ArrayLike, *, tmin: float = 0.0, tmax: float = 1.0
    ) -> np.ndarray:
        """Map event times (spikes, licks, button presses ...) onto the aligned clock.

        Parameters
        ----------
        trials : array_like of int
            Trial index of every event.
        times : array_like of float
            Event times within their trial, in the same units as ``tmin`` / ``tmax``.
        tmin, tmax : float
            Times of the first and last sample of the data the model was fitted on
            (for binned data, the first and last bin centers). The defaults treat
            ``times`` as normalized trial time.

        Returns
        -------
        ndarray, same shape as ``times``
            Aligned event times in the same units.
        """
        return self._map_events(trials, times, tmin, tmax, inverse=False)

    def inverse_transform_events(
        self, trials: ArrayLike, times: ArrayLike, *, tmin: float = 0.0, tmax: float = 1.0
    ) -> np.ndarray:
        """Map aligned (template) times back onto each trial's own clock.

        Useful for locating, in every trial, a moment identified on the template.
        Arguments are as for :meth:`transform_events`.
        """
        return self._map_events(trials, times, tmin, tmax, inverse=True)

    def _map_events(self, trials, times, tmin, tmax, *, inverse: bool) -> np.ndarray:
        self._check_fitted()
        trials = as_index_array(trials, "trials")
        times = np.asarray(times, dtype=float)
        trials, times = np.broadcast_arrays(trials, times)
        if trials.size and (trials.min() < 0 or trials.max() >= self.n_trials_):
            raise IndexError(f"trial indices must be in [0, {self.n_trials_})")
        span = float(tmax) - float(tmin)
        if not span > 0:
            raise ValueError("tmax must be greater than tmin")
        k = trials.ravel()
        x, y = self.x_knots_[k], self.y_knots_[k]
        if inverse:
            x, y = y, x
        u = (times.ravel() - tmin) / span
        out = evaluate_warps(x, y, u[:, np.newaxis])[:, 0]
        return (tmin + span * out).reshape(times.shape)


class PiecewiseWarping(_BaseWarping):
    """Align trials with piecewise-linear time warps.

    Each trial gets an increasing warp made of ``n_knots + 1`` straight
    segments, so different parts of a trial can be shifted, stretched or
    compressed by different amounts. ``n_knots=0`` gives linear warps (shift
    plus uniform stretch); :class:`LinearWarping` is a shortcut for that.

    Parameters
    ----------
    n_knots : int, default 1
        Number of interior knots (bends) per warp. Both the time and the value
        of every knot are fitted per trial.
    max_lag : float, default 0.3
        Hard limit on ``|ω(t) - t|``: no part of a trial is moved by more than
        this fraction of the trial duration.
    max_stretch : float, default 3.0
        Hard limit on local stretching: every segment's slope lies in
        ``[1 / max_stretch, max_stretch]``.
    warp_reg : float, default 1.0
        Weight of the penalty ``∫ (ω(t) - t)^2 dt`` that pulls warps toward the
        identity, relative to the data term (residual variance / data
        variance). Larger values give smaller, more conservative warps.
    smoothness : float, default 0.01
        Template roughness penalty, expressed as a smoothing timescale in units
        of the trial duration (0 turns it off). A smooth template also gives
        the warp search a smoother objective.
    l2_reg : float, default 0.0
        Ridge penalty shrinking the template toward each feature's mean.
    n_bins : int, optional
        Number of template samples. Defaults to the number of timepoints.
    random_state : int, numpy.random.Generator or None
        Seed for the randomized warp search.

    Attributes
    ----------
    template_ : ndarray, shape (n_bins, n_features)
        The learned template, sampled evenly over template time [0, 1].
    x_knots_, y_knots_ : ndarray, shape (n_trials, n_knots + 2)
        Knots of each trial's warp (trial time -> template time).
    loss_history_ : ndarray
        Objective after initialization and after each iteration.
    n_trials_, n_timepoints_, n_features_ : int
        Shape of the training data.
    """

    def __init__(
        self,
        n_knots: int = 1,
        *,
        max_lag: float = 0.3,
        max_stretch: float = 3.0,
        warp_reg: float = 1.0,
        smoothness: float = 0.01,
        l2_reg: float = 0.0,
        n_bins: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            max_lag=max_lag,
            warp_reg=warp_reg,
            smoothness=smoothness,
            l2_reg=l2_reg,
            n_bins=n_bins,
        )
        self.n_knots = n_knots
        self.max_stretch = max_stretch
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        *,
        n_iter: int = 20,
        n_proposals: int = 10,
        n_candidates: int = 100,
        init: _BaseWarping | None = None,
        verbose: bool = False,
    ) -> Self:
        """Learn the template and one warp per trial.

        Parameters
        ----------
        X : array_like, shape (n_trials, n_timepoints, n_features) or (n_trials, n_timepoints)
            Trials of equal length. NaN marks missing samples (e.g. trials shorter
            than the window, padded with NaN); they are ignored by the fit.
        n_iter : int, default 20
            Number of warp/template update rounds.
        n_proposals : int, default 10
            Local-search proposals per trial and round.
        n_candidates : int, default 100
            Random candidate warps scored against every trial each round, on top
            of a fixed grid of shifts and stretches.
        init : fitted trialwarp model, optional
            Start from that model's warps (e.g. a :class:`ShiftWarping` or
            :class:`LinearWarping` fit) instead of the identity.
        verbose : bool, default False
            Print the loss after every round.

        Returns
        -------
        self
        """
        rng = check_random_state(self.random_state)
        return self._fit(
            X,
            n_iter=n_iter,
            init=init,
            verbose=verbose,
            rng=rng,
            n_proposals=n_proposals,
            n_candidates=n_candidates,
        )

    def _validate_params(self) -> None:
        super()._validate_params()
        if int(self.n_knots) != self.n_knots or self.n_knots < 0:
            raise ValueError(f"n_knots must be a non-negative integer, got {self.n_knots}")
        if not self.max_stretch >= 1.0:
            raise ValueError(f"max_stretch must be >= 1, got {self.max_stretch}")

    @property
    def _n_points(self) -> int:
        return int(self.n_knots) + 2

    def _slope_bounds(self) -> tuple[float, float]:
        return 1.0 / self.max_stretch, float(self.max_stretch)

    def _setup_search(self) -> None:
        n_time = self._t.size
        lag = self.max_lag
        # shifts every ~2 samples (at most 31 of them) times a few stretch factors
        n_shift = int(min(31, 2 * math.ceil(lag * (n_time - 1) / 2) + 1))
        centers = np.linspace(-lag, lag, n_shift)
        lo, hi = self._slope_bounds()
        lo, hi = max(lo, 1.0 - 2.0 * lag), min(hi, 1.0 + 2.0 * lag)
        slopes = np.unique(np.concatenate([np.geomspace(lo, hi, 9), [1.0]]))
        c, s = (a.ravel() for a in np.meshgrid(centers, slopes, indexing="ij"))
        grid = np.linspace(0.0, 1.0, self._n_points)
        y = 0.5 + c[:, np.newaxis] + s[:, np.newaxis] * (grid - 0.5)
        x = np.broadcast_to(grid, y.shape).copy()
        ok = self._is_valid(x, y)
        self._grid_x, self._grid_y = x[ok], y[ok]
        spacing = centers[1] - centers[0] if n_shift > 1 else lag
        self._step0 = spacing / 2.0
        self._step_min = 0.05 / (n_time - 1)
        self._step_max = lag

    def _candidate_knots(self, rng, n_candidates):
        if rng is None or n_candidates <= 0:
            return self._grid_x, self._grid_y
        xr, yr = random_knots(
            n_candidates,
            int(self.n_knots),
            max_lag=self.max_lag,
            max_stretch=self.max_stretch,
            rng=rng,
        )
        ok = self._is_valid(xr, yr)
        return np.concatenate([self._grid_x, xr[ok]]), np.concatenate([self._grid_y, yr[ok]])

    def _propose(self, x, y, step, rng):
        n_trials, n_points = y.shape
        kind = rng.integers(0, 4, n_trials)
        eps = rng.standard_normal(n_trials) * step
        xn, yn = x.copy(), y.copy()

        sel = kind == 0  # shift the whole warp
        yn[sel] += eps[sel, np.newaxis]

        sel = np.flatnonzero(kind == 1)  # stretch about a random pivot
        if sel.size:
            pivot = rng.uniform(0.0, 1.0, sel.size)[:, np.newaxis]
            span = np.maximum(pivot, 1.0 - pivot)
            yn[sel] += eps[sel, np.newaxis] * (x[sel] - pivot) / span

        sel = np.flatnonzero(kind == 2)  # move a single knot
        if sel.size:
            j = rng.integers(0, n_points, sel.size)
            yn[sel, j] += eps[sel]
            inner = (j > 0) & (j < n_points - 1)
            xn[sel[inner], j[inner]] += rng.standard_normal(inner.sum()) * step[sel[inner]]

        sel = np.flatnonzero(kind == 3)  # jitter every knot independently
        if sel.size:
            yn[sel] += rng.standard_normal((sel.size, n_points)) * step[sel, np.newaxis]
            if n_points > 2:
                xn[sel, 1:-1] += (
                    rng.standard_normal((sel.size, n_points - 2)) * step[sel, np.newaxis]
                )
        return xn, yn

    def _project(self, t, values):
        n_trials = values.shape[0]
        grid = np.linspace(0.0, 1.0, self._n_points)
        x = np.tile(grid, (n_trials, 1))
        y = np.stack([np.interp(grid, t, v) for v in values])
        bad = ~self._is_valid(x, y)
        if bad.any():
            y[bad] = self._line_knots(t, values[bad], grid)
            bad = ~self._is_valid(x, y)
            y[bad] = x[bad]
        return x, y


class LinearWarping(PiecewiseWarping):
    """Align trials with linear time warps ``ω(t) = a + b t`` (shift plus uniform stretch).

    Equivalent to ``PiecewiseWarping(n_knots=0)``; see :class:`PiecewiseWarping`
    for the parameters, attributes and :meth:`~PiecewiseWarping.fit` options.
    """

    def __init__(
        self,
        *,
        max_lag: float = 0.3,
        max_stretch: float = 3.0,
        warp_reg: float = 1.0,
        smoothness: float = 0.01,
        l2_reg: float = 0.0,
        n_bins: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            0,
            max_lag=max_lag,
            max_stretch=max_stretch,
            warp_reg=warp_reg,
            smoothness=smoothness,
            l2_reg=l2_reg,
            n_bins=n_bins,
            random_state=random_state,
        )

    @property
    def intercepts_(self) -> np.ndarray:
        """``a`` in ``ω(t) = a + b t``: template time at the start of each trial."""
        self._check_fitted()
        return self.y_knots_[:, 0].copy()

    @property
    def slopes_(self) -> np.ndarray:
        """``b`` in ``ω(t) = a + b t``; above 1 means the trial runs faster than the template."""
        self._check_fitted()
        return self.y_knots_[:, 1] - self.y_knots_[:, 0]


class ShiftWarping(_BaseWarping):
    """Align trials by shifting each one in time, ``ω(t) = t + shift``.

    Shifts are found by an exhaustive search over a grid of sub-sample
    resolution, so the warp step is exact up to that resolution and needs no
    random search.

    Parameters
    ----------
    max_lag : float, default 0.2
        Largest shift considered, as a fraction of the trial duration.
    warp_reg : float, default 1.0
        Weight of the penalty ``shift**2`` (see :class:`PiecewiseWarping`).
    smoothness : float, default 0.01
        Template smoothing timescale, as a fraction of the trial duration.
    l2_reg : float, default 0.0
        Ridge penalty shrinking the template toward each feature's mean.
    n_bins : int, optional
        Number of template samples. Defaults to the number of timepoints.
    resolution : float, default 0.5
        Spacing of the shift grid, in samples.

    Attributes
    ----------
    shifts_ : ndarray, shape (n_trials,)
        Fitted shifts in normalized time. A positive shift means the trial's
        activity happens earlier than in the template. Multiply by
        ``n_timepoints_ - 1`` for samples.
    template_, x_knots_, y_knots_, loss_history_, n_trials_, n_timepoints_, n_features_
        As for :class:`PiecewiseWarping`.
    """

    def __init__(
        self,
        *,
        max_lag: float = 0.2,
        warp_reg: float = 1.0,
        smoothness: float = 0.01,
        l2_reg: float = 0.0,
        n_bins: int | None = None,
        resolution: float = 0.5,
    ) -> None:
        super().__init__(
            max_lag=max_lag,
            warp_reg=warp_reg,
            smoothness=smoothness,
            l2_reg=l2_reg,
            n_bins=n_bins,
        )
        self.resolution = resolution

    def fit(
        self,
        X: ArrayLike,
        *,
        n_iter: int = 10,
        init: _BaseWarping | None = None,
        verbose: bool = False,
    ) -> Self:
        """Learn the template and one shift per trial.

        Parameters
        ----------
        X : array_like, shape (n_trials, n_timepoints, n_features) or (n_trials, n_timepoints)
            Trials of equal length; NaN marks missing samples.
        n_iter : int, default 10
            Number of shift/template update rounds.
        init : fitted trialwarp model, optional
            Start from that model's warps instead of zero shifts.
        verbose : bool, default False
            Print the loss after every round.

        Returns
        -------
        self
        """
        return self._fit(
            X,
            n_iter=n_iter,
            init=init,
            verbose=verbose,
            rng=None,
            n_proposals=0,
            n_candidates=0,
        )

    def _validate_params(self) -> None:
        super()._validate_params()
        if not self.resolution > 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")

    @property
    def shifts_(self) -> np.ndarray:
        """Shift of each trial in normalized time; positive means the trial runs early."""
        self._check_fitted()
        return self.y_knots_[:, 0].copy()

    @property
    def _n_points(self) -> int:
        return 2

    def _slope_bounds(self) -> tuple[float, float]:
        return 1.0, 1.0

    def _setup_search(self) -> None:
        spacing = self.resolution / (self._t.size - 1)
        m = math.floor(self.max_lag / spacing + 1e-9)
        shifts = np.arange(-m, m + 1) * spacing
        self._grid_x = np.tile([0.0, 1.0], (shifts.size, 1))
        self._grid_y = np.column_stack([shifts, 1.0 + shifts])

    def _candidate_knots(self, rng, n_candidates):
        return self._grid_x, self._grid_y

    def _project(self, t, values):
        shift = np.clip((values - t).mean(axis=1), -self.max_lag, self.max_lag)
        x = np.tile([0.0, 1.0], (len(shift), 1))
        return x, np.column_stack([shift, 1.0 + shift])
