"""trialwarp: align repeated trials of multi-dimensional time series with time warping.

Fit a model to an array of trials shaped ``(n_trials, n_timepoints, n_features)``
to learn a shared template plus one warping function per trial, then use the
warps to align the data (or other data recorded on the same trials)::

    import trialwarp as tw

    model = tw.PiecewiseWarping(n_knots=1).fit(X)
    aligned = model.transform(X)

Models: :class:`ShiftWarping`, :class:`LinearWarping`, :class:`PiecewiseWarping`.
Plots live in :mod:`trialwarp.plotting` (needs matplotlib).
"""

from .datasets import SpikeTrials, SyntheticTrials, make_spike_trials, make_warped_trials
from .models import LinearWarping, PiecewiseWarping, ShiftWarping
from .neural import bin_spikes, smooth_trials
from .validation import HeldoutScore, clone, heldout_feature_score, trial_consistency
from .warps import evaluate_warps, invert_warps

__version__ = "0.1.0"

__all__ = [
    "HeldoutScore",
    "LinearWarping",
    "PiecewiseWarping",
    "ShiftWarping",
    "SpikeTrials",
    "SyntheticTrials",
    "__version__",
    "bin_spikes",
    "clone",
    "evaluate_warps",
    "heldout_feature_score",
    "invert_warps",
    "make_spike_trials",
    "make_warped_trials",
    "smooth_trials",
    "trial_consistency",
]
