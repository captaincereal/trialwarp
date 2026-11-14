import numpy as np
import pytest

import trialwarp as tw
from trialwarp.models import _BaseWarping
from trialwarp.warps import random_knots

FAST = {"n_iter": 8}


@pytest.fixture(scope="module")
def warped():
    return tw.make_warped_trials(
        n_trials=60, n_timepoints=80, n_features=6, n_knots=1, noise=0.3, random_state=0
    )


def oracle(data):
    """A model object carrying the true warps, to align data perfectly."""
    model = tw.PiecewiseWarping(n_knots=data.x_knots.shape[1] - 2)
    model.x_knots_, model.y_knots_ = data.x_knots, data.y_knots
    model.n_trials_, model.n_timepoints_ = data.data.shape[:2]
    return model


@pytest.mark.parametrize(
    ("model", "true_knots"),
    [
        (tw.LinearWarping(random_state=0), 0),
        (tw.PiecewiseWarping(n_knots=1, random_state=0), 1),
        (tw.PiecewiseWarping(n_knots=1, random_state=0), 0),
    ],
    ids=["linear", "piecewise", "piecewise-on-linear"],
)
def test_alignment_recovers_structure(model, true_knots):
    data = tw.make_warped_trials(
        n_trials=60,
        n_timepoints=80,
        n_features=6,
        n_knots=true_knots,
        noise=0.3,
        random_state=0,
    )
    model.fit(data.data, **FAST)
    raw = tw.trial_consistency(data.clean)
    aligned = tw.trial_consistency(model.transform(data.clean))
    assert aligned > 0.97
    assert aligned > raw + 0.3


def test_piecewise_warps_match_truth(warped):
    model = tw.PiecewiseWarping(n_knots=1, random_state=0).fit(warped.data, **FAST)
    t = warped.time
    err = model.warping_functions(t) - warped.warping_functions(t)
    err -= err.mean(axis=0)  # the template's own clock is arbitrary
    rms_samples = np.sqrt(np.mean(err**2)) * (len(t) - 1)
    assert rms_samples < 2.0


def test_shift_recovers_known_shifts():
    rng = np.random.default_rng(0)
    n_time = 100
    t = np.linspace(0.0, 1.0, n_time)
    shifts = rng.integers(-8, 9, 40) / (n_time - 1)
    centers = np.array([0.3, 0.5, 0.7])
    # trial k at time t shows the template at t + shift_k
    X = np.exp(-0.5 * ((t[None, :, None] + shifts[:, None, None] - centers) / 0.04) ** 2)
    model = tw.ShiftWarping(warp_reg=0.0).fit(X + 0.05 * rng.standard_normal(X.shape))
    est = model.shifts_ - model.shifts_.mean()
    np.testing.assert_allclose(est, shifts - shifts.mean(), atol=0.75 / (n_time - 1))


@pytest.mark.parametrize(
    "model",
    [
        tw.ShiftWarping(),
        tw.LinearWarping(random_state=1),
        tw.PiecewiseWarping(n_knots=2, random_state=1),
    ],
    ids=["shift", "linear", "piecewise"],
)
def test_loss_never_increases(warped, model):
    model.fit(warped.data, **({"n_iter": 6}))
    history = model.loss_history_
    assert history.shape == (7,)
    assert np.all(np.diff(history) <= 1e-9 * history[0])


@pytest.mark.parametrize("n_knots", [0, 1, 3])
def test_fitted_warps_satisfy_constraints(warped, n_knots):
    model = tw.PiecewiseWarping(n_knots=n_knots, max_lag=0.15, max_stretch=1.5, random_state=2)
    model.fit(warped.data, **FAST)
    x, y = model.x_knots_, model.y_knots_
    assert x.shape == (60, n_knots + 2)
    np.testing.assert_array_equal(x[:, 0], 0.0)
    np.testing.assert_array_equal(x[:, -1], 1.0)
    slopes = np.diff(y, axis=1) / np.diff(x, axis=1)
    assert np.all(slopes >= 1 / 1.5 - 1e-9) and np.all(slopes <= 1.5 + 1e-9)
    assert np.all(np.abs(y - x) <= 0.15 + 1e-9)


def test_identical_trials_need_no_warping():
    t = np.linspace(0.0, 1.0, 60)
    trial = np.column_stack([np.sin(2 * np.pi * t), np.exp(-(((t - 0.4) / 0.1) ** 2))])
    X = np.repeat(trial[None], 10, axis=0)
    model = tw.PiecewiseWarping(n_knots=1, random_state=0).fit(X, **FAST)
    np.testing.assert_allclose(model.warping_functions(t), np.tile(t, (10, 1)), atol=1e-9)
    np.testing.assert_allclose(model.predict(), X, atol=0.02)


def test_transform_with_true_warps_aligns_perfectly():
    # finely sampled, so linear interpolation of the narrow bumps is accurate
    data = tw.make_warped_trials(
        n_trials=20, n_timepoints=400, n_features=4, noise=0.0, random_state=1
    )
    aligned = oracle(data).transform(data.clean)
    inside = np.isfinite(aligned[..., 0])
    assert inside.mean() > 0.8
    template = np.broadcast_to(data.template, aligned.shape)
    np.testing.assert_allclose(aligned[inside], template[inside], atol=0.01)


def test_transform_shapes_and_fill(warped):
    model = tw.LinearWarping(random_state=0).fit(warped.data, n_iter=3)
    assert model.transform(warped.data).shape == warped.data.shape
    two_d = model.transform(warped.data[:, :, 0])
    assert two_d.shape == warped.data.shape[:2]
    resampled = model.transform(np.repeat(warped.data, 2, axis=1))  # other sampling rate
    assert resampled.shape == (60, 160, 6)
    filled = model.transform(warped.data, fill_value=-7.0)
    assert np.isfinite(filled).all()
    with pytest.raises(ValueError, match="trials"):
        model.transform(warped.data[:10])


def test_events_round_trip_and_units(warped):
    model = tw.PiecewiseWarping(n_knots=1, random_state=0).fit(warped.data, n_iter=3)
    rng = np.random.default_rng(0)
    trials = rng.integers(0, 60, 200)
    times = rng.uniform(0.0, 1.0, 200)
    aligned = model.transform_events(trials, times)
    np.testing.assert_allclose(model.inverse_transform_events(trials, aligned), times, atol=1e-12)
    in_seconds = model.transform_events(trials, 2.0 + 3.0 * times, tmin=2.0, tmax=5.0)
    np.testing.assert_allclose(in_seconds, 2.0 + 3.0 * aligned, atol=1e-12)
    with pytest.raises(IndexError):
        model.transform_events([60], [0.5])
    with pytest.raises(TypeError):
        model.transform_events([1.5], [0.5])


def test_event_times_follow_aligned_data():
    # one sharp peak per trial; after alignment it must sit where transform_events says
    rng = np.random.default_rng(4)
    n_time = 120
    t = np.linspace(0.0, 1.0, n_time)
    x, y = random_knots(30, 1, max_lag=0.1, max_stretch=1.4, rng=rng)
    peak = 0.5  # template time of the peak
    trial_peak = tw.invert_warps(x, y, [peak])[:, 0]
    X = np.exp(-0.5 * ((t[None, :] - trial_peak[:, None]) / 0.01) ** 2)
    model = tw.PiecewiseWarping(n_knots=1, random_state=0)
    model.x_knots_, model.y_knots_, model.n_trials_, model.n_timepoints_ = x, y, 30, n_time
    aligned_peak = t[np.nanargmax(model.transform(X), axis=1)]
    expected = model.transform_events(np.arange(30), trial_peak)
    np.testing.assert_allclose(aligned_peak, expected, atol=1.0 / (n_time - 1))


def test_missing_data():
    data = tw.make_warped_trials(
        n_trials=40, n_timepoints=70, n_features=5, noise=0.3, random_state=5
    )
    X = data.data.copy()
    rng = np.random.default_rng(0)
    X[rng.random(X.shape) < 0.1] = np.nan  # scattered missing entries
    lengths = rng.integers(55, 71, 40)
    ragged = data.data.copy()
    for k, n in enumerate(lengths):
        ragged[k, n:] = np.nan  # shorter trials padded with NaN
    for arr in (X, ragged):
        model = tw.PiecewiseWarping(n_knots=1, random_state=0).fit(arr, **FAST)
        assert np.isfinite(model.template_).all()
        assert np.all(np.diff(model.loss_history_) <= 1e-9 * model.loss_history_[0])
        assert tw.trial_consistency(model.transform(data.clean)) > 0.9


def test_reproducible_with_seed(warped):
    a = tw.PiecewiseWarping(n_knots=1, random_state=7).fit(warped.data, n_iter=3)
    b = tw.PiecewiseWarping(n_knots=1, random_state=7).fit(warped.data, n_iter=3)
    np.testing.assert_array_equal(a.y_knots_, b.y_knots_)
    np.testing.assert_array_equal(a.template_, b.template_)


def test_init_from_simpler_model(warped):
    linear = tw.LinearWarping(random_state=0).fit(warped.data, n_iter=5)
    start = tw.PiecewiseWarping(n_knots=2, random_state=0).fit(warped.data, n_iter=0, init=linear)
    t = np.linspace(0.0, 1.0, 50)
    np.testing.assert_allclose(start.warping_functions(t), linear.warping_functions(t), atol=1e-9)
    shift = tw.ShiftWarping().fit(warped.data, n_iter=0, init=linear)
    assert np.all(np.abs(shift.shifts_) <= shift.max_lag)
    with pytest.raises(TypeError):
        tw.LinearWarping().fit(warped.data, init="identity")


def test_projected_search_scores_are_exact():
    """Scores on the template's principal axes differ from full scores by a per-trial
    constant, so they rank warps identically."""
    # more features than template samples, so the template's rank is below n_features
    data = tw.make_warped_trials(
        n_trials=12, n_timepoints=40, n_features=90, noise=0.5, random_state=3
    )
    model = tw.PiecewiseWarping(n_knots=1, random_state=0).fit(data.data, n_iter=2)
    # re-create the fitting state, then compare scores of two sets of warps
    model._prepare(data.data)
    model._x, model._y = model.x_knots_.copy(), model.y_knots_.copy()
    model._update_template()
    reduced = model._search_space()
    assert reduced.X.shape[2] <= 40
    full = model._full_space()
    rng = np.random.default_rng(0)
    x2, y2 = random_knots(12, 1, max_lag=0.2, max_stretch=2.0, rng=rng)
    d_full = model._trial_objective(full, x2, y2) - model._trial_objective(
        full, model._x, model._y
    )
    d_reduced = model._trial_objective(reduced, x2, y2) - model._trial_objective(
        reduced, model._x, model._y
    )
    np.testing.assert_allclose(d_reduced, d_full, rtol=1e-6, atol=1e-10)


def test_params_clone_and_repr():
    model = tw.PiecewiseWarping(n_knots=2, warp_reg=0.5, random_state=3)
    params = model.get_params()
    assert params["n_knots"] == 2 and params["warp_reg"] == 0.5
    copy = tw.clone(model)
    assert copy is not model and copy.get_params() == params
    assert "n_knots=2" in repr(model)
    model.set_params(max_lag=0.2)
    assert model.max_lag == 0.2
    with pytest.raises(ValueError):
        model.set_params(nonsense=1)
    assert "n_knots" not in tw.LinearWarping().get_params()


@pytest.mark.parametrize(
    "bad",
    [
        {"n_knots": -1},
        {"n_knots": 1.5},
        {"max_lag": 0.0},
        {"max_stretch": 0.5},
        {"warp_reg": -1.0},
        {"n_bins": 2},
    ],
)
def test_invalid_parameters_raise(warped, bad):
    with pytest.raises(ValueError):
        tw.PiecewiseWarping(**bad).fit(warped.data, n_iter=1)


def test_unfitted_and_bad_input_raise(warped):
    with pytest.raises(RuntimeError, match="not fitted"):
        tw.LinearWarping().transform(warped.data)
    with pytest.raises(ValueError, match="shape"):
        tw.LinearWarping().fit(np.zeros(10))
    with pytest.raises(ValueError, match="two trials"):
        tw.LinearWarping().fit(np.zeros((1, 10, 2)))
    with pytest.raises(ValueError, match="finite"):
        tw.LinearWarping().fit(np.full((3, 10, 2), np.nan))


def test_template_resolution_can_differ(warped):
    model = tw.PiecewiseWarping(n_knots=1, n_bins=40, random_state=0).fit(warped.data, **FAST)
    assert model.template_.shape == (40, 6)
    assert model.predict().shape == warped.data.shape
    assert tw.trial_consistency(model.transform(warped.clean)) > 0.95


def test_base_class_is_internal():
    assert not hasattr(tw, "_BaseWarping")
    assert issubclass(tw.ShiftWarping, _BaseWarping)
