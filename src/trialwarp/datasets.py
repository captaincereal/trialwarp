"""Synthetic data with known warps, for examples, tests and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._utils import check_random_state
from .warps import evaluate_warps, identity_knots, invert_warps, random_knots

__all__ = ["SpikeTrials", "SyntheticTrials", "make_spike_trials", "make_warped_trials"]


def _true_warps(
    rng: np.random.Generator,
    n_trials: int,
    n_knots: int,
    max_lag: float,
    max_stretch: float,
    log_slope_sd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Random ground-truth warps; ``max_lag=0`` gives identity warps (a null control)."""
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative, got {max_lag}")
    if max_lag == 0:
        return identity_knots(n_trials, n_knots)
    while True:
        x, y = random_knots(
            n_trials,
            n_knots,
            max_lag=max_lag,
            max_stretch=max_stretch,
            rng=rng,
            log_slope_sd=log_slope_sd,
        )
        if len(x) == n_trials:
            return x, y
        log_slope_sd /= 2.0  # warp shapes too varied to fit within max_lag: make them gentler


@dataclass(frozen=True)
class SyntheticTrials:
    """Noisy warped trials together with the ground truth that produced them.

    Attributes
    ----------
    data : ndarray, shape (n_trials, n_timepoints, n_features)
        Observations: ``clean`` plus Gaussian noise.
    clean : ndarray, same shape
        The template warped by each trial's true warp, without noise.
    template : ndarray, shape (n_timepoints, n_features)
        Ground-truth template on the sample grid.
    x_knots, y_knots : ndarray, shape (n_trials, n_knots + 2)
        True warps, mapping trial time to template time like a fitted model's.
    noise : float
        Standard deviation of the added noise.
    """

    data: np.ndarray
    clean: np.ndarray
    template: np.ndarray
    x_knots: np.ndarray
    y_knots: np.ndarray
    noise: float

    @property
    def time(self) -> np.ndarray:
        """Normalized sample times, ``linspace(0, 1, n_timepoints)``."""
        return np.linspace(0.0, 1.0, self.data.shape[1])

    def warping_functions(self, t: ArrayLike | None = None) -> np.ndarray:
        """True warps evaluated at normalized trial times ``t`` (default: sample times)."""
        t = self.time if t is None else np.atleast_1d(t)
        return evaluate_warps(self.x_knots, self.y_knots, t)


def _bump_template(rng: np.random.Generator, n_features: int):
    """Smooth template: every feature has a main bump (together they form a
    sequence across the trial) and possibly a smaller positive or negative one."""
    main = np.linspace(0.22, 0.78, n_features) + rng.normal(0.0, 0.015, n_features)
    centers = np.column_stack([main, rng.uniform(0.2, 0.8, n_features)])
    widths = rng.uniform(0.02, 0.04, (n_features, 2))
    amps = np.column_stack(
        [rng.uniform(0.8, 1.2, n_features), rng.choice([0.0, 0.5, -0.5], n_features)]
    )

    def template(tau: np.ndarray) -> np.ndarray:
        z = (tau[..., np.newaxis, np.newaxis] - centers) / widths
        return np.sum(amps * np.exp(-0.5 * z**2), axis=-1)

    return template


def make_warped_trials(
    n_trials: int = 100,
    n_timepoints: int = 150,
    n_features: int = 10,
    *,
    n_knots: int = 1,
    max_lag: float = 0.12,
    max_stretch: float = 1.5,
    noise: float = 0.5,
    random_state: int | np.random.Generator | None = None,
) -> SyntheticTrials:
    """Generate trials of a multi-dimensional signal with random piecewise-linear warps.

    The template is a set of Gaussian bumps (amplitude ~1) whose peaks are
    spread across the trial. Each trial reads the template through its own
    random warp with ``n_knots`` interior knots, displaced from the identity by
    at most ``max_lag``, and gets independent Gaussian noise of standard
    deviation ``noise``. ``n_knots=0`` gives linear (shift-and-stretch) warps;
    ``max_lag=0`` gives unwarped trials, a null control for checking that a
    model does not invent structure.
    """
    rng = check_random_state(random_state)
    template = _bump_template(rng, n_features)
    t = np.linspace(0.0, 1.0, n_timepoints)
    x, y = _true_warps(rng, n_trials, n_knots, max_lag, max_stretch, log_slope_sd=0.12)
    clean = template(evaluate_warps(x, y, t))
    data = clean + noise * rng.standard_normal(clean.shape)
    return SyntheticTrials(
        data=data, clean=clean, template=template(t), x_knots=x, y_knots=y, noise=noise
    )


@dataclass(frozen=True)
class SpikeTrials:
    """Simulated spike trains with known warps, in the flat format of most spike sorters.

    Attributes
    ----------
    trials, times, neurons : ndarray, shape (n_spikes,)
        Trial index, time within the trial (seconds, in ``[0, duration)``) and
        neuron index of every spike.
    n_trials, n_neurons : int
    duration : float
        Trial length in seconds.
    x_knots, y_knots : ndarray
        True warps in normalized time (trial time -> template time).
    peak_times : ndarray, shape (n_neurons,)
        When each neuron fires most on the template clock, in seconds.
    event_times : ndarray, shape (n_trials,)
        Time of a behavioral event (say, movement onset) in every trial. It
        happens at ``event_template_time`` on the template clock, so its timing
        varies across trials exactly as the neural activity does.
    event_template_time : float
    """

    trials: np.ndarray
    times: np.ndarray
    neurons: np.ndarray
    n_trials: int
    n_neurons: int
    duration: float
    x_knots: np.ndarray
    y_knots: np.ndarray
    peak_times: np.ndarray
    event_times: np.ndarray
    event_template_time: float


def make_spike_trials(
    n_trials: int = 120,
    n_neurons: int = 40,
    *,
    duration: float = 1.0,
    n_knots: int = 1,
    max_lag: float = 0.1,
    max_stretch: float = 1.6,
    baseline_rate: float = 4.0,
    peak_rate: float = 40.0,
    width: float = 0.04,
    event_template_time: float = 0.5,
    dt: float = 0.001,
    random_state: int | np.random.Generator | None = None,
) -> SpikeTrials:
    """Simulate Poisson spike trains from a population that fires in a sequence.

    Neuron ``n`` has a Gaussian rate bump (``width`` as a fraction of the trial)
    at a preferred moment of the template, spread so the population tiles the
    trial. Each trial runs the sequence through its own piecewise-linear warp,
    so peaks move and stretch together from trial to trial, as they do when an
    animal's behavior varies in timing.
    """
    rng = check_random_state(random_state)
    x, y = _true_warps(rng, n_trials, n_knots, max_lag, max_stretch, log_slope_sd=0.15)
    peaks = np.sort(rng.uniform(0.12, 0.88, n_neurons))
    gains = peak_rate * rng.uniform(0.5, 1.0, n_neurons)
    n_steps = round(duration / dt)
    t = (np.arange(n_steps) + 0.5) / n_steps  # normalized time at each step's center
    tau = evaluate_warps(x, y, t)  # (n_trials, n_steps)
    z = (tau[:, :, np.newaxis] - peaks) / width
    rate = baseline_rate + gains * np.exp(-0.5 * z**2)
    counts = rng.poisson(rate * (duration / n_steps))
    k, i, n = np.nonzero(counts)
    reps = counts[k, i, n]
    k, i, n = np.repeat(k, reps), np.repeat(i, reps), np.repeat(n, reps)
    times = (i + rng.uniform(0.0, 1.0, i.size)) * (duration / n_steps)
    order = np.lexsort((times, n, k))
    events = invert_warps(x, y, [event_template_time / duration])[:, 0] * duration
    return SpikeTrials(
        trials=k[order],
        times=times[order],
        neurons=n[order],
        n_trials=n_trials,
        n_neurons=n_neurons,
        duration=duration,
        x_knots=x,
        y_knots=y,
        peak_times=peaks * duration,
        event_times=events,
        event_template_time=event_template_time,
    )
