"""Neural-style data: align spike trains from a population that fires in a sequence.

We simulate 40 neurons recorded over 120 trials. Each trial runs the same
activity sequence, but on its own clock, as when an animal's reaction time
varies from trial to trial. A behavioral event (say, movement onset) is locked
to the sequence, so its time also varies.

The usual pipeline:

1. bin spike times into a (trials, time bins, neurons) array and smooth it;
2. fit a warping model on the smoothed rates;
3. apply the warps to the raw *spike times* and to the behavioral event times,
   so rasters can be drawn on the aligned clock without any binning.

Run from the repository root:

    python examples/02_neural_spikes.py [--no-show]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import trialwarp as tw
from trialwarp import plotting

FIGURES = Path(__file__).resolve().parent / "figures"


def main(show: bool = True) -> None:
    FIGURES.mkdir(exist_ok=True)

    # Spikes in the flat format most spike sorters export: one entry per spike.
    spikes = tw.make_spike_trials(n_trials=120, n_neurons=40, random_state=1)
    n_spikes, n_neurons, n_trials = spikes.times.size, spikes.n_neurons, spikes.n_trials
    print(f"{n_spikes} spikes from {n_neurons} neurons in {n_trials} trials")

    # 1. Bin (10 ms bins over the 1 s trial) and smooth into firing rates.
    counts, centers = tw.bin_spikes(
        spikes.trials,
        spikes.times,
        spikes.neurons,
        n_trials=spikes.n_trials,
        n_units=spikes.n_neurons,
        n_bins=100,
        tmax=spikes.duration,
    )
    bin_width = centers[1] - centers[0]
    rates = tw.smooth_trials(counts, sigma=2.0) / bin_width  # spikes per second

    # 2. Fit. One knot lets the early and late parts of a trial warp differently.
    model = tw.PiecewiseWarping(n_knots=1, random_state=0).fit(rates)

    # 3. Warp spike times and event times. The model's normalized clock runs from
    #    the first to the last bin center, so those are tmin and tmax.
    clock = {"tmin": centers[0], "tmax": centers[-1]}
    aligned_spikes = model.transform_events(spikes.trials, spikes.times, **clock)
    trials = np.arange(spikes.n_trials)
    aligned_events = model.transform_events(trials, spikes.event_times, **clock)
    print(
        f"movement-onset jitter (std): raw {1e3 * spikes.event_times.std():.1f} ms, "
        f"aligned {1e3 * aligned_events.std():.1f} ms"
    )

    # Does warping generalize? Fit warps on 4/5 of the neurons, test on the rest.
    score = tw.heldout_feature_score(
        tw.PiecewiseWarping(n_knots=1, random_state=0), rates, random_state=0
    )
    print(score)

    # Figures ---------------------------------------------------------------
    with plotting.style():
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, layout="constrained")
    for col, neuron in enumerate([6, 20, 33]):
        mine = spikes.neurons == neuron
        plotting.plot_raster(
            spikes.trials[mine],
            spikes.times[mine],
            ax=axes[0, col],
            events=spikes.event_times,
            title=f"Neuron {neuron}, raw",
        )
        plotting.plot_raster(
            spikes.trials[mine],
            aligned_spikes[mine],
            ax=axes[1, col],
            events=aligned_events,
            title=f"Neuron {neuron}, aligned",
        )
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    for ax in axes[0]:
        ax.set_xlabel("")
    fig.suptitle(
        "Spike rasters before and after warping (dots: movement onset)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.savefig(FIGURES / "02_rasters.png", dpi=150)

    plot_population(rates, model.transform(rates), centers)
    plt.savefig(FIGURES / "02_population.png", dpi=150)

    print(f"figures saved to {FIGURES}")
    if show:
        plt.show()


def plot_population(rates, aligned, centers):
    """Trial-averaged rate of every neuron (rows, sorted by peak time), before and after."""
    plain = rates.mean(axis=0).T
    warped = np.nanmean(aligned, axis=0).T
    order = np.argsort(np.argmax(warped, axis=1))
    # scale each neuron to its own peak so all rows use the full color range
    peak = np.maximum(warped.max(axis=1, keepdims=True), 1e-12)
    with plotting.style():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        common = {"time": centers, "vmin": 0.0, "vmax": 1.0, "cmap": plotting.SEQUENTIAL}
        plotting.plot_trials(
            (plain / peak)[order],
            ax=axes[0],
            colorbar=False,
            title="Plain trial average",
            **common,
        )
        plotting.plot_trials(
            (warped / peak)[order],
            ax=axes[1],
            label="rate / peak rate",
            title="Average of aligned trials",
            **common,
        )
        for ax in axes:
            ax.set_ylabel("neuron (sorted by peak time)")
            ax.set_xlabel("time (s)")
        axes[1].set_ylabel("")
        fig.suptitle(
            "Population sequence: alignment restores sharp, ordered peaks",
            x=0.01,
            ha="left",
            fontsize=11,
        )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="save figures without showing")
    main(show=not parser.parse_args().no_show)
