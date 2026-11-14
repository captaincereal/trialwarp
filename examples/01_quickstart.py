"""Quickstart: learn a template and per-trial warps, and compare with plain averaging.

We simulate 100 noisy trials of a 10-dimensional signal in which every trial
runs on its own clock (a random piecewise-linear time warp), fit warping
models, and compare three estimates of the underlying pattern:

* the plain trial average, which smears features whose timing varies;
* the average of the aligned trials;
* the model's template.

Run from the repository root:

    python examples/01_quickstart.py            # show figures
    python examples/01_quickstart.py --no-show  # only save them to examples/figures/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import trialwarp as tw
from trialwarp import plotting

FIGURES = Path(__file__).resolve().parent / "figures"


def r_squared(estimate, truth):
    """Fraction of the ground truth's variance captured by an estimate."""
    return 1.0 - np.sum((estimate - truth) ** 2) / np.sum((truth - truth.mean(axis=0)) ** 2)


def main(show: bool = True) -> None:
    FIGURES.mkdir(exist_ok=True)

    # 1. Data: (n_trials, n_timepoints, n_features), with the truth kept for comparison.
    data = tw.make_warped_trials(
        n_trials=100, n_timepoints=150, n_features=10, n_knots=1, noise=0.5, random_state=0
    )
    X = data.data
    print(f"data: {X.shape[0]} trials x {X.shape[1]} timepoints x {X.shape[2]} features")

    # 2. Fit. Linear warps (shift + stretch) first, then refine with one-knot
    #    piecewise warps that start from the linear solution.
    linear = tw.LinearWarping(random_state=0).fit(X)
    model = tw.PiecewiseWarping(n_knots=1, random_state=0).fit(X, init=linear, verbose=True)

    # 3. Align the trials: each one is resampled onto the template's clock.
    aligned = model.transform(X)

    # 4. How close does each estimate get to the true pattern?
    plain = X.mean(axis=0)
    aligned_mean = np.nanmean(aligned, axis=0)
    print("\nR^2 against the true template (1 = perfect):")
    print(f"  plain average             {r_squared(plain, data.template):.3f}")
    print(f"  average of aligned trials {r_squared(aligned_mean, data.template):.3f}")
    print(f"  model template            {r_squared(model.template_, data.template):.3f}")

    # 5. Figures.
    feature = 4
    fig = plotting.plot_alignment(X, aligned, feature=feature, truth=data.template)
    fig.suptitle(f"Feature {feature}: raw vs. aligned trials", x=0.01, ha="left", fontsize=11)
    fig.savefig(FIGURES / "01_alignment.png", dpi=150)

    plot_all_features(data, model)
    plt.savefig(FIGURES / "01_all_features.png", dpi=150)

    plot_warp_recovery(data, model)
    plt.savefig(FIGURES / "01_warps.png", dpi=150)

    print(f"\nfigures saved to {FIGURES}")
    if show:
        plt.show()


def plot_warp_recovery(data, model):
    """True vs. estimated warps, a per-trial comparison of the two, and the loss."""
    with plotting.style():
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
    plotting.plot_warps(data, ax=axes[0, 0], title="True warps")
    plotting.plot_warps(model, ax=axes[0, 1], title="Estimated warps")
    axes[0, 1].sharey(axes[0, 0])

    # displacement of every trial at five moments, truth vs. estimate
    t = np.linspace(0.1, 0.9, 5)
    true = (data.warping_functions(t) - t).ravel()
    est = (model.warping_functions(t) - t).ravel()
    with plotting.style():
        ax = axes[1, 0]
        lim = 1.1 * max(np.abs(true).max(), np.abs(est).max())
        ax.plot([-lim, lim], [-lim, lim], color=plotting.AXIS, lw=0.8, zorder=1)
        ax.scatter(true, est, s=10, color=plotting.COLORS["blue"], alpha=0.5, lw=0)
        ax.set(
            xlim=(-lim, lim),
            ylim=(-lim, lim),
            aspect="equal",
            xlabel="true displacement",
            ylabel="estimated displacement",
        )
        corr = np.corrcoef(true, est)[0, 1]
        ax.set_title(f"Per-trial displacement, r = {corr:.2f}")
    plotting.plot_loss(model, ax=axes[1, 1])
    return fig


def plot_all_features(data, model):
    """Small multiples: plain average vs. model template vs. truth, for every feature."""
    t = np.linspace(0.0, 1.0, data.data.shape[1])
    plain = data.data.mean(axis=0)
    with plotting.style():
        fig, axes = plt.subplots(
            2, 5, figsize=(12, 4.4), sharex=True, sharey=True, layout="constrained"
        )
        for n, ax in enumerate(axes.flat):
            ax.plot(
                t, data.template[:, n], color=plotting.INK_SECONDARY, lw=1.2, label="ground truth"
            )
            ax.plot(t, plain[:, n], color=plotting.COLORS["orange"], label="plain average")
            ax.plot(
                t, model.template_[:, n], color=plotting.COLORS["blue"], label="warped template"
            )
            ax.set_title(f"feature {n}")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        for ax in axes[1]:
            ax.set_xlabel("time")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper right", ncols=3)
        fig.suptitle(
            "Plain averaging blurs every feature; the warped template keeps them sharp",
            x=0.01,
            ha="left",
            fontsize=11,
        )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="save figures without showing")
    main(show=not parser.parse_args().no_show)
