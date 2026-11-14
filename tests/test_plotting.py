import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import trialwarp as tw  # noqa: E402
from trialwarp import plotting  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    data = tw.make_warped_trials(n_trials=20, n_timepoints=40, n_features=3, random_state=0)
    model = tw.PiecewiseWarping(n_knots=1, random_state=0).fit(data.data, n_iter=2)
    return data, model


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_alignment_figure(fitted):
    data, model = fitted
    fig = plotting.plot_alignment(
        data.data,
        model.transform(data.data),
        feature=1,
        template=model.template_,
        truth=data.template,
        time=(-0.5, 1.5),
    )
    assert len(fig.axes) == 4  # three panels and a colorbar


def test_single_panels(fitted):
    data, model = fitted
    assert plotting.plot_trials(np.abs(data.data), title="rates", label="Hz") is not None
    assert plotting.plot_warps(model, kind="warp", trials=[0, 1, 2]) is not None
    assert plotting.plot_warps(model, time=(0.0, 2.0)) is not None
    assert plotting.plot_loss(model) is not None
    with pytest.raises(ValueError):
        plotting.plot_warps(model, kind="nope")


def test_raster(fitted):
    spikes = tw.make_spike_trials(n_trials=8, n_neurons=2, random_state=0)
    one = spikes.neurons == 0
    ax = plotting.plot_raster(
        spikes.trials[one],
        spikes.times[one],
        order=np.argsort(spikes.event_times),
        events=spikes.event_times,
    )
    assert ax.get_ylim() == (7.5, -0.5)


def test_style_context_restores_rcparams():
    before = matplotlib.rcParams["axes.facecolor"]
    with plotting.style():
        assert matplotlib.rcParams["axes.facecolor"] == plotting.SURFACE
    assert matplotlib.rcParams["axes.facecolor"] == before
