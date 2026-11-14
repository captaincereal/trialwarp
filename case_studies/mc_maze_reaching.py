"""Case study: recovering reach timing from motor cortex activity alone.

Data: MC_Maze (Churchland & Kaufman; Neural Latents Benchmark). A macaque makes
delayed reaches through a maze while ~160 neurons in primary motor and dorsal
premotor cortex are recorded. Published on DANDI under CC-BY-4.0.

The problem: reaction times vary from trial to trial (here, 260-506 ms for the
middle 90% of trials). Aligned to the go cue, the only event the experimenter
controls, movement-related activity is smeared across trials.

The test: fit ``PiecewiseWarping`` to the neural activity only, then check the
warps against two behavioral signals the model never saw:

1. the measured movement onset of every trial, and
2. the hand's speed profile, recorded at 1 kHz.

Run from the repository root (downloads the data once, with a checksum check):

    pip install -e ".[case-studies]"
    python case_studies/mc_maze_reaching.py                    # 149 MB, ~3 min
    python case_studies/mc_maze_reaching.py --dataset small    # 29 MB, quicker
    python case_studies/mc_maze_reaching.py --crossval         # adds held-out-neuron validation
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import trialwarp as tw
from trialwarp import plotting

FIGURES = Path(__file__).resolve().parent / "figures"

# Published (immutable) DANDI versions, with the SHA-256 of each file.
DATASETS = {
    "small": {
        "dandiset": "000140",
        "version": "0.220113.0408",
        "asset": "7821971e-c6a4-4568-8773-1bfa205c13f8",
        "sha256": "dcaf3a524e2b2f65f163ee3b07789b8474bdfc6ca66098bc542ab93dff489884",
        "bytes": 29_207_528,
    },
    "medium": {
        "dandiset": "000139",
        "version": "0.220113.0408",
        "asset": "7ef450a8-8684-42e2-8598-cd38ca2b2e50",
        "sha256": "3852799f85f662f1cacadb0d758ba8fa0dc6cca318f46c9b131978adf1baffb6",
        "bytes": 76_604_764,
    },
    "large": {
        "dandiset": "000138",
        "version": "0.220113.0407",
        "asset": "e67b57b2-e9ad-4d95-b9e3-1262997360dc",
        "sha256": "1188ddf5b822dd9ac49afb8916f4b42a85867613cf7f169d0b04a1d1aeecd700",
        "bytes": 148_590_536,
    },
}


# ---------------------------------------------------------------------------
# data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, data_dir: Path) -> Path:
    """Fetch one MC_Maze training file from DANDI (once) and verify its checksum."""
    info = DATASETS[name]
    path = data_dir / f"mc_maze_{name}" / "train.nwb"
    if path.exists() and sha256(path) == info["sha256"]:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://api.dandiarchive.org/api/assets/{info['asset']}/download/"
    print(f"downloading MC_Maze_{name} ({info['bytes'] / 1e6:.0f} MB) from DANDI to {path}")
    partial = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if sha256(partial) != info["sha256"]:
        partial.unlink()
        raise RuntimeError("checksum mismatch: the download is incomplete or the file changed")
    partial.replace(path)
    return path


@dataclass
class Session:
    """Trials aligned to the go cue. Times are in seconds relative to the go cue."""

    rates: np.ndarray  # (trials, bins, neurons): sqrt of smoothed firing rate
    bin_centers: np.ndarray  # (bins,)
    spike_trial: np.ndarray  # one entry per spike in the window
    spike_time: np.ndarray
    spike_unit: np.ndarray  # column of `rates` the spike belongs to
    speed: np.ndarray  # (trials, samples): hand speed in mm/s, 1 kHz
    speed_time: np.ndarray  # (samples,)
    reaction_time: np.ndarray  # (trials,): movement onset - go cue
    condition: np.ndarray  # (trials,): reach condition
    unit_ids: np.ndarray  # (neurons,): NWB unit id of each column of `rates`

    @property
    def clock(self) -> dict[str, float]:
        """``tmin``/``tmax`` for transform_events: the first and last bin centers."""
        return {"tmin": float(self.bin_centers[0]), "tmax": float(self.bin_centers[-1])}


def load_session(
    path: Path,
    pre: float = 0.2,
    post: float = 0.9,
    bin_size: float = 0.01,
    sigma_bins: float = 2.0,
    min_rate: float = 1.0,
) -> Session:
    """Spikes, hand speed and trial events in a window around the go cue.

    Trials whose target appeared less than ``pre`` seconds before the go cue are
    dropped, so the window never contains the target onset. Neurons firing below
    ``min_rate`` Hz in the window are dropped. Rates are smoothed with a Gaussian
    of ``sigma_bins`` bins and square-root transformed, which evens out the
    influence of low- and high-rate neurons.
    """
    with h5py.File(path, "r") as f:
        trials = f["intervals/trials"]
        go = trials["go_cue_time"][:]
        onset = trials["move_onset_time"][:]
        target_on = trials["target_on_time"][:]
        condition = trials["trial_type"][:]
        spike_times = f["units/spike_times"][:]
        ends = f["units/spike_times_index"][:]
        unit_ids = f["units/id"][:]
        vel_t = f["processing/behavior/hand_vel/timestamps"][:]
        # stored in mm/s (the file's unit attribute says m/s, but peaks are ~700)
        vel = f["processing/behavior/hand_vel/data"][:]

    keep = np.flatnonzero(go - target_on >= pre)
    unit = np.repeat(np.arange(ends.size), np.diff(ends, prepend=0))
    order = np.argsort(spike_times, kind="stable")
    spike_times, unit = spike_times[order], unit[order]

    k_list, t_list, u_list = [], [], []
    for i, k in enumerate(keep):
        lo, hi = np.searchsorted(spike_times, [go[k] - pre, go[k] + post])
        k_list.append(np.full(hi - lo, i))
        t_list.append(spike_times[lo:hi] - go[k])
        u_list.append(unit[lo:hi])
    spike_trial, spike_time, spike_unit = (np.concatenate(a) for a in (k_list, t_list, u_list))

    n_bins = round((pre + post) / bin_size)
    counts, centers = tw.bin_spikes(
        spike_trial,
        spike_time,
        spike_unit,
        n_trials=keep.size,
        n_units=ends.size,
        n_bins=n_bins,
        tmin=-pre,
        tmax=post,
    )
    rate = tw.smooth_trials(counts, sigma_bins) / bin_size
    active = np.flatnonzero(rate.mean(axis=(0, 1)) >= min_rate)
    column = np.full(ends.size, -1)
    column[active] = np.arange(active.size)
    in_use = column[spike_unit] >= 0

    # hand speed on the same clock as the bins, at 1 ms resolution
    speed_time = np.linspace(centers[0], centers[-1], round((centers[-1] - centers[0]) * 1000) + 1)
    hand_speed = np.hypot(vel[:, 0], vel[:, 1])
    speed = np.stack([np.interp(go[k] + speed_time, vel_t, hand_speed) for k in keep])

    return Session(
        rates=np.sqrt(rate[:, :, active]),
        bin_centers=centers,
        spike_trial=spike_trial[in_use],
        spike_time=spike_time[in_use],
        spike_unit=column[spike_unit[in_use]],
        speed=speed,
        speed_time=speed_time,
        reaction_time=onset[keep] - go[keep],
        condition=condition[keep],
        unit_ids=unit_ids[active],
    )


# ---------------------------------------------------------------------------
# validation against behavior


def predicted_onsets(model, session: Session) -> tuple[np.ndarray, np.ndarray]:
    """Each trial's movement onset as predicted by its warp, plus measured onsets on
    the template clock.

    One number is calibrated from behavior: where movement starts on the template
    clock (the median of the measured onsets mapped onto it). Every per-trial
    prediction then comes from the neural warps alone.
    """
    trials = np.arange(len(session.reaction_time))
    on_template = model.transform_events(trials, session.reaction_time, **session.clock)
    marker = np.full(trials.size, np.median(on_template))
    return model.inverse_transform_events(trials, marker, **session.clock), on_template


def onset_aligned_speed(session: Session) -> np.ndarray:
    """Hand speed realigned on the measured movement onset (the behavioral 'oracle')."""
    shift = session.reaction_time - np.median(session.reaction_time)
    t = session.speed_time
    return np.stack(
        [
            np.interp(t + d, t, s, left=np.nan, right=np.nan)
            for s, d in zip(session.speed, shift, strict=True)
        ]
    )


def iqr(values: np.ndarray) -> float:
    return float(np.subtract(*np.percentile(values, [75, 25])))


def evaluate(model, session: Session) -> dict[str, float]:
    predicted, on_template = predicted_onsets(model, session)
    measured = session.reaction_time
    return {
        "r": float(np.corrcoef(predicted, measured)[0, 1]),
        "median_error_ms": 1e3 * float(np.median(np.abs(predicted - measured))),
        "onset_iqr_ms": 1e3 * iqr(on_template),
        "speed_consistency": tw.trial_consistency(model.transform(session.speed)),
    }


# ---------------------------------------------------------------------------
# figures


def plot_overview(model, session: Session, scores: dict[str, float]):
    """Hero figure: hand speed before/after neural-only warping, and onset prediction."""
    ms = 1e3 * session.speed_time
    warped = model.transform(session.speed)
    predicted, _ = predicted_onsets(model, session)
    with plotting.style():
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(13, 3.8),
            layout="constrained",
            gridspec_kw={"width_ratios": [1, 1, 0.8]},
        )
        panels = [
            (session.speed, "Aligned to the go cue", "time from go cue (ms)"),
            (warped, "Warped by neural activity alone", "aligned time (ms)"),
        ]
        for ax, (values, title, xlabel) in zip(axes[:2], panels, strict=True):
            ax.plot(ms, values.T, color=plotting.COLORS["blue"], lw=0.6, alpha=0.06)
            ax.plot(ms, np.nanmedian(values, axis=0), color=plotting.INK, lw=1.8, label="median")
            ax.axvline(0.0, color=plotting.AXIS, lw=0.8, zorder=0)
            ax.set(title=title, xlabel=xlabel, xlim=(ms[0], ms[-1]))
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        axes[0].set_ylabel("hand speed (mm/s)")
        axes[1].sharey(axes[0])
        axes[1].legend(loc="upper right")

        ax = axes[2]
        x, y = 1e3 * session.reaction_time, 1e3 * predicted
        lo, hi = np.percentile(np.concatenate([x, y]), [0.5, 99.5])
        lo, hi = lo - 20, hi + 20
        ax.plot([lo, hi], [lo, hi], color=plotting.AXIS, lw=0.8, zorder=0)
        ax.scatter(x, y, s=12, color=plotting.COLORS["blue"], alpha=0.55, lw=0)
        ax.set(
            xlim=(lo, hi),
            ylim=(lo, hi),
            aspect="equal",
            xlabel="measured onset, ms after go cue",
            ylabel="predicted from neurons (ms)",
        )
        ax.set_title(f"Reaction time from spikes, r = {scores['r']:.2f}")
        fig.suptitle(
            f"{session.rates.shape[0]} reaches, {session.rates.shape[2]} motor-cortex neurons: "
            "warps learned from spikes line up the hand movements they never saw",
            x=0.01,
            ha="left",
            fontsize=11,
        )
    return fig


def pick_neurons(model, session: Session, n: int = 3) -> np.ndarray:
    """Neurons whose trial average sharpens most when the trials are warped."""
    aligned = model.transform(session.rates)
    gain = np.nanvar(np.nanmean(aligned, axis=0), axis=0) - np.var(
        session.rates.mean(axis=0), axis=0
    )
    return np.argsort(gain)[::-1][:n]


def plot_rasters(model, session: Session, neurons: np.ndarray):
    """Rasters before/after warping, trials sorted by measured reaction time (display only)."""
    order = np.argsort(session.reaction_time)
    trials = np.arange(len(order))
    on_template = model.transform_events(trials, session.reaction_time, **session.clock)
    warped_time = model.transform_events(session.spike_trial, session.spike_time, **session.clock)
    with plotting.style():
        fig, axes = plt.subplots(
            2, len(neurons), figsize=(4.2 * len(neurons), 6.6), sharex=True, layout="constrained"
        )
    for col, unit in enumerate(neurons):
        mine = session.spike_unit == unit
        for row, (times, events, label) in enumerate(
            [
                (session.spike_time[mine], session.reaction_time, "aligned to go cue"),
                (warped_time[mine], on_template, "warped"),
            ]
        ):
            ax = axes[row, col]
            plotting.plot_raster(
                session.spike_trial[mine],
                1e3 * times,
                ax=ax,
                order=order,
                events=1e3 * events,
                size=4,
                title=f"Unit {session.unit_ids[unit]}, {label}",
            )
            ax.set_xlim(-250, 1050)
            ax.set_xlabel("time from go cue (ms)" if row else "")
            ax.set_ylabel("trial (sorted by reaction time)" if col == 0 else "")
    fig.suptitle(
        "Spikes before and after warping (orange: measured movement onset)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    return fig


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="trialwarp case study on MC_Maze reaching data")
    parser.add_argument("--dataset", choices=list(DATASETS), default="large")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".cache" / "trialwarp")
    parser.add_argument("--crossval", action="store_true", help="also score held-out neurons")
    parser.add_argument("--no-show", action="store_true", help="save figures without showing")
    args = parser.parse_args()

    session = load_session(download(args.dataset, args.data_dir))
    n_trials, n_bins, n_neurons = session.rates.shape
    rt = 1e3 * session.reaction_time
    print(
        f"{n_trials} trials x {n_bins} bins (10 ms) x {n_neurons} neurons; "
        f"reaction time median {np.median(rt):.0f} ms, IQR {iqr(rt):.0f} ms"
    )

    raw_speed = tw.trial_consistency(session.speed)
    oracle_speed = tw.trial_consistency(onset_aligned_speed(session))
    rows = [
        ("no warping (go-cue aligned)", None, None, iqr(rt), raw_speed),
        ("aligned on measured onset (oracle)", None, None, 0.0, oracle_speed),
    ]
    models = {
        "ShiftWarping": tw.ShiftWarping(max_lag=0.3),
        "LinearWarping": tw.LinearWarping(random_state=0),
        "PiecewiseWarping(n_knots=1)": tw.PiecewiseWarping(n_knots=1, random_state=0),
        "PiecewiseWarping(n_knots=2)": tw.PiecewiseWarping(n_knots=2, random_state=0),
    }
    fitted = {}
    for name, model in models.items():
        start = time.perf_counter()
        fitted[name] = model.fit(session.rates)
        s = evaluate(model, session)
        rows.append(
            (name, s["r"], s["median_error_ms"], s["onset_iqr_ms"], s["speed_consistency"])
        )
        print(f"  fitted {name} in {time.perf_counter() - start:.0f} s")

    print(
        "\n| alignment | r (predicted vs. measured onset) | median error (ms) "
        "| onset IQR (ms) | hand-speed consistency |"
    )
    print("|---|---|---|---|---|")
    for name, r, err, spread, cons in rows:
        r_txt = "-" if r is None else f"{r:.2f}"
        err_txt = "-" if err is None else f"{err:.0f}"
        print(f"| {name} | {r_txt} | {err_txt} | {spread:.0f} | {cons:.3f} |")

    if args.crossval:
        score = tw.heldout_feature_score(
            tw.PiecewiseWarping(n_knots=1, random_state=0), session.rates, random_state=0
        )
        print(f"\nheld-out neurons, PiecewiseWarping(n_knots=1): {score}")

    featured = fitted["PiecewiseWarping(n_knots=1)"]
    FIGURES.mkdir(exist_ok=True)
    plot_overview(featured, session, evaluate(featured, session))
    plt.savefig(FIGURES / "mc_maze_overview.png", dpi=150)
    plot_rasters(featured, session, pick_neurons(featured, session))
    plt.savefig(FIGURES / "mc_maze_rasters.png", dpi=150)
    print(f"\nfigures saved to {FIGURES}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
