"""Plots for inspecting an alignment (requires matplotlib: ``pip install trialwarp[plot]``).

Every function draws into ``ax`` when given one and otherwise makes its own
figure, and returns the axes (or the figure, for multi-panel plots) so you
can keep styling it with ordinary matplotlib calls.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from numpy.typing import ArrayLike

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator
except ImportError as err:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "trialwarp.plotting needs matplotlib; install it with `pip install trialwarp[plot]`"
    ) from err

from ._utils import as_index_array, as_trials_array

__all__ = [
    "COLORS",
    "plot_alignment",
    "plot_averages",
    "plot_loss",
    "plot_raster",
    "plot_trials",
    "plot_warps",
    "style",
]

#: Categorical colors in a fixed, colorblind-checked order. Assign them in this
#: order (blue, orange, aqua, ...) rather than cycling or picking by taste.
COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#6250d6",
    "red": "#e34948",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

#: One-hue (blue) map for non-negative data such as firing rates.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "trialwarp_sequential", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
).with_extremes(bad=SURFACE)
#: Blue-gray-red map for signed data, centered on zero.
DIVERGING = LinearSegmentedColormap.from_list(
    "trialwarp_diverging", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9c2b2b"]
).with_extremes(bad=SURFACE)

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": 9,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
    "font.size": 9,
    "text.color": INK,
}


@contextmanager
def style() -> Iterator[None]:
    """Context manager applying trialwarp's quiet plot style (hairline axes, muted ink).

    The plotting functions use it internally; wrap your own figure code in it to
    match them.
    """
    with plt.rc_context(_RC):
        yield


def _new_axes(ax, figsize):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    return ax


def _feature(X: ArrayLike, feature: int) -> np.ndarray:
    X3, _ = as_trials_array(X)
    return X3[:, :, feature]


def _time_axis(time: ArrayLike | None, n: int) -> np.ndarray:
    if time is None:
        return np.linspace(0.0, 1.0, n)
    time = np.asarray(time, dtype=float)
    if time.size == 2:
        return np.linspace(time[0], time[1], n)
    if time.size != n:
        raise ValueError(f"time must have {n} entries (or be a (start, stop) pair)")
    return time


def _color_scale(values: np.ndarray, cmap, vmin, vmax):
    if cmap is not None:
        return cmap, vmin, vmax
    lo, hi = np.nanpercentile(values, [1.0, 99.0])
    if lo >= 0:
        return SEQUENTIAL, 0.0 if vmin is None else vmin, hi if vmax is None else vmax
    lim = max(abs(lo), abs(hi))
    return DIVERGING, -lim if vmin is None else vmin, lim if vmax is None else vmax


def plot_trials(
    X: ArrayLike,
    *,
    feature: int = 0,
    ax=None,
    time: ArrayLike | None = None,
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    colorbar: bool = True,
    label: str | None = None,
):
    """Heatmap of one feature, trials as rows and time as columns.

    Non-negative data get a one-hue blue scale starting at zero; signed data get
    a blue-gray-red scale centered on zero. NaN (e.g. outside the window after
    alignment) is left blank.

    Parameters
    ----------
    X : array_like, shape (n_trials, n_timepoints[, n_features])
    feature : int
        Which feature to show.
    time : array_like, optional
        Sample times, or a ``(start, stop)`` pair. Defaults to normalized time.
    cmap, vmin, vmax
        Override the automatic color scale.
    title, label : str, optional
        Axes title and colorbar label.
    """
    with style():
        values = _feature(X, feature)
        n_trials, n_time = values.shape
        ax = _new_axes(ax, (4.0, 3.2))
        time = _time_axis(time, n_time)
        cmap, vmin, vmax = _color_scale(values, cmap, vmin, vmax)
        half = (time[-1] - time[0]) / (2 * (n_time - 1))
        image = ax.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=(time[0] - half, time[-1] + half, n_trials - 0.5, -0.5),
        )
        ax.set_xlabel("time")
        ax.set_ylabel("trial")
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        if title:
            ax.set_title(title)
        if colorbar:
            bar = ax.figure.colorbar(image, ax=ax, fraction=0.05, pad=0.02, aspect=30)
            bar.outline.set_visible(False)
            bar.ax.tick_params(length=0)
            if label:
                bar.set_label(label)
    return ax


def plot_averages(
    X: ArrayLike,
    aligned: ArrayLike,
    *,
    feature: int = 0,
    ax=None,
    time: ArrayLike | None = None,
    template: ArrayLike | None = None,
    truth: ArrayLike | None = None,
    title: str | None = "Trial average",
):
    """Plain trial average next to the average after alignment.

    Parameters
    ----------
    X, aligned : array_like, shape (n_trials, n_timepoints[, n_features])
        Raw and aligned trials (``aligned = model.transform(X)``).
    template : array_like, shape (n_bins[, n_features]), optional
        A model's ``template_``, drawn over the same time span.
    truth : array_like, shape (n_samples[, n_features]), optional
        Ground truth, when known (synthetic data).
    """
    with style():
        raw = _feature(X, feature)
        warped = _feature(aligned, feature)
        ax = _new_axes(ax, (4.8, 3.2))
        time = _time_axis(time, raw.shape[1])

        def span(values):
            values = np.asarray(values, dtype=float)
            values = values[:, feature] if values.ndim == 2 else values
            return np.linspace(time[0], time[-1], values.size), values

        if truth is not None:
            ax.plot(*span(truth), color=INK_SECONDARY, lw=1.2, label="ground truth", zorder=4)
        ax.plot(time, np.nanmean(raw, axis=0), color=COLORS["orange"], label="plain average")
        ax.plot(time, np.nanmean(warped, axis=0), color=COLORS["blue"], label="aligned average")
        if template is not None:
            ax.plot(*span(template), color=COLORS["aqua"], label="model template")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_xlabel("time")
        ax.legend(loc="best")
        if title:
            ax.set_title(title)
    return ax


def plot_alignment(
    X: ArrayLike,
    aligned: ArrayLike,
    *,
    feature: int = 0,
    time: ArrayLike | None = None,
    template: ArrayLike | None = None,
    truth: ArrayLike | None = None,
    figsize: tuple[float, float] = (12.0, 3.4),
):
    """Three panels for one feature: raw trials, aligned trials, and their averages.

    Both heatmaps share one color scale so they can be compared directly. See
    :func:`plot_averages` for ``template`` and ``truth``. Returns the figure.
    """
    with style():
        raw = _feature(X, feature)
        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize,
            layout="constrained",
            gridspec_kw={"width_ratios": [1, 1.1, 1.2]},
        )
        cmap, vmin, vmax = _color_scale(raw, None, None, None)
        common = {"feature": 0, "time": time, "cmap": cmap, "vmin": vmin, "vmax": vmax}
        plot_trials(raw, ax=axes[0], title="Raw trials", colorbar=False, **common)
        plot_trials(_feature(aligned, feature), ax=axes[1], title="Aligned trials", **common)
        axes[1].set_ylabel("")
        plot_averages(
            X, aligned, feature=feature, ax=axes[2], time=time, template=template, truth=truth
        )
    return fig


def plot_warps(
    model,
    *,
    ax=None,
    kind: str = "displacement",
    trials: ArrayLike | None = None,
    time: ArrayLike | None = None,
    color: str = COLORS["blue"],
    alpha: float | None = None,
    title: str | None = "Warping functions",
):
    """Draw each trial's fitted warp.

    Parameters
    ----------
    model : fitted trialwarp model
        Or anything else with a ``warping_functions(t)`` method, such as the
        ground truth returned by :func:`~trialwarp.make_warped_trials`.
    kind : {"displacement", "warp"}
        ``"displacement"`` (default) plots ``ω(t) - t``, where shifts and
        stretches are easy to read; ``"warp"`` plots ``ω(t)`` against the
        identity line.
    trials : array_like of int, optional
        Subset of trials to draw.
    time : (start, stop), optional
        Express both axes in these units instead of normalized time.
    """
    if kind not in ("displacement", "warp"):
        raise ValueError("kind must be 'displacement' or 'warp'")
    with style():
        ax = _new_axes(ax, (4.2, 3.2))
        t = np.linspace(0.0, 1.0, 201)
        values = model.warping_functions(t)
        if trials is not None:
            values = values[as_index_array(trials, "trials")]
        start, stop = (0.0, 1.0) if time is None else (float(time[0]), float(time[-1]))
        scale = stop - start
        if kind == "displacement":
            curves = (values - t) * scale
            ax.axhline(0.0, color=AXIS, lw=0.8, zorder=1)
            ax.set_ylabel("template time - trial time")
        else:
            curves = start + values * scale
            ax.plot([start, stop], [start, stop], color=AXIS, lw=0.8, zorder=1)
            ax.set_ylabel("template time")
        if alpha is None:
            alpha = float(np.clip(6.0 / max(len(curves), 1), 0.12, 0.9))
        ax.plot(start + t * scale, curves.T, color=color, lw=0.9, alpha=alpha)
        ax.set_xlabel("trial time")
        ax.set_xlim(start, stop)
        if title:
            ax.set_title(title)
    return ax


def plot_loss(model, *, ax=None, title: str | None = "Training loss"):
    """Loss after initialization (iteration 0) and after every iteration."""
    with style():
        ax = _new_axes(ax, (4.0, 2.8))
        history = model.loss_history_
        ax.plot(np.arange(history.size), history, color=COLORS["blue"], marker="o", ms=3.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        if title:
            ax.set_title(title)
    return ax


def plot_raster(
    trials: ArrayLike,
    times: ArrayLike,
    *,
    ax=None,
    order: ArrayLike | None = None,
    events: ArrayLike | None = None,
    color: str = INK,
    event_color: str = COLORS["orange"],
    size: float = 6.0,
    title: str | None = None,
):
    """Spike raster: one row per trial, one tick per spike.

    Parameters
    ----------
    trials, times : array_like, shape (n_spikes,)
        Trial index and time of every spike (filter to one neuron first).
    order : array_like of int, optional
        Trial indices from top to bottom; defaults to trial order.
    events : array_like, shape (n_trials,), optional
        One event time per trial, marked with a dot (e.g. movement onset).
    """
    trials = as_index_array(trials, "trials").ravel()
    times = np.asarray(times, dtype=float).ravel()
    n_rows = int(trials.max()) + 1 if trials.size else 0
    if events is not None:
        n_rows = max(n_rows, np.size(events))
    if order is not None:
        order = as_index_array(order, "order")
        n_rows = order.size
    rank = np.arange(n_rows) if order is None else np.argsort(order)
    with style():
        ax = _new_axes(ax, (4.0, 3.2))
        ax.scatter(times, rank[trials], s=size, marker="|", color=color, linewidths=0.7)
        if events is not None:
            events = np.asarray(events, dtype=float)
            ax.scatter(
                events,
                rank[np.arange(events.size)],
                s=7,
                color=event_color,
                lw=0,
                zorder=3,
                label="event",
            )
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_xlabel("time")
        ax.set_ylabel("trial")
        ax.spines["left"].set_visible(False)
        if title:
            ax.set_title(title)
    return ax
