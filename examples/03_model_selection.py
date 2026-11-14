"""Choosing a model, and checking that warping is not just fitting noise.

Warping always lowers the training loss, even on pure noise, so the loss
cannot tell you whether the warps are real. ``heldout_feature_score`` asks a
sharper question: do warps fitted on some features improve predictions of
features the model never saw (for neural data, held-out neurons)?

We score four models on two data sets built from the same template:

* "warped": every trial has a random one-knot warp;
* "null": identical generator with the warps switched off (``max_lag=0``).

On the warped data the flexible models should win; on the null data no model
should help, which is the check that the method is not inventing structure.

Run from the repository root:

    python examples/03_model_selection.py [--no-show]
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
    common = {"n_trials": 80, "n_timepoints": 100, "n_features": 12, "noise": 0.6}
    datasets = {
        "warped data": tw.make_warped_trials(n_knots=1, random_state=2, **common).data,
        "null data (no warps)": tw.make_warped_trials(max_lag=0.0, random_state=2, **common).data,
    }
    models = {
        "shift": tw.ShiftWarping(),
        "linear": tw.LinearWarping(random_state=0),
        "piecewise\n1 knot": tw.PiecewiseWarping(n_knots=1, random_state=0),
        "piecewise\n2 knots": tw.PiecewiseWarping(n_knots=2, random_state=0),
    }

    results = {}
    for data_name, X in datasets.items():
        print(f"\n{data_name}")
        for model_name, model in models.items():
            score = tw.heldout_feature_score(model, X, n_folds=4, random_state=0)
            results[data_name, model_name] = score.improvement
            print(f"  {model_name.replace(chr(10), ' '):20s} {score}")

    plot_scores(results, list(datasets), list(models))
    plt.savefig(FIGURES / "03_model_selection.png", dpi=150)
    print(f"\nfigure saved to {FIGURES}")
    if show:
        plt.show()


def plot_scores(results, dataset_names, model_names):
    """Held-out improvement (mean and spread over folds) per model and data set."""
    colors = [plotting.COLORS["blue"], plotting.COLORS["orange"]]
    x = np.arange(len(model_names))
    with plotting.style():
        fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
        ax.axhline(0.0, color=plotting.AXIS, lw=0.8, zorder=1)
        for i, (name, color) in enumerate(zip(dataset_names, colors, strict=True)):
            values = [results[name, m] for m in model_names]
            means = np.array([v.mean() for v in values])
            spread = np.array([v.std() for v in values])
            offset = (i - 0.5) * 0.18
            ax.errorbar(
                x + offset,
                means,
                yerr=spread,
                fmt="o",
                ms=6,
                color=color,
                elinewidth=1.2,
                capsize=0,
                label=name,
            )
            ax.annotate(
                name,
                (x[-1] + offset, means[-1]),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=plotting.INK_SECONDARY,
            )
        ax.set_xticks(x, model_names)
        ax.set_xlim(-0.5, len(model_names) + 0.4)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_ylabel("held-out R$^2$ gain from warping")
        ax.legend(loc="upper left")
        ax.set_title("Warps help on held-out features only when the timing variability is real")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="save figures without showing")
    main(show=not parser.parse_args().no_show)
