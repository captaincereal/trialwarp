import numpy as np
import pytest

from trialwarp.warps import (
    evaluate_warps,
    identity_knots,
    invert_warps,
    random_knots,
    warp_penalty,
)


@pytest.fixture
def knots():
    rng = np.random.default_rng(0)
    return random_knots(25, 3, max_lag=0.2, max_stretch=2.0, rng=rng)


def test_evaluate_matches_np_interp(knots):
    x, y = knots
    t = np.linspace(0.0, 1.0, 77)
    expected = np.stack([np.interp(t, xk, yk) for xk, yk in zip(x, y, strict=True)])
    np.testing.assert_allclose(evaluate_warps(x, y, t), expected, atol=1e-12)


def test_evaluate_accepts_per_warp_times(knots):
    x, y = knots
    t = np.random.default_rng(1).uniform(0.0, 1.0, (len(x), 9))
    expected = np.stack([np.interp(tk, xk, yk) for tk, xk, yk in zip(t, x, y, strict=True)])
    np.testing.assert_allclose(evaluate_warps(x, y, t), expected, atol=1e-12)


def test_evaluate_extends_end_segments_linearly():
    x = np.array([[0.0, 0.5, 1.0]])
    y = np.array([[0.1, 0.4, 1.2]])
    out = evaluate_warps(x, y, [-0.5, 1.5])
    np.testing.assert_allclose(out, [[0.1 - 0.5 * 0.6, 1.2 + 0.5 * 1.6]])


def test_inverse_undoes_warp(knots):
    x, y = knots
    t = np.linspace(0.0, 1.0, 50)
    round_trip = invert_warps(x, y, evaluate_warps(x, y, t))
    np.testing.assert_allclose(round_trip, np.broadcast_to(t, round_trip.shape), atol=1e-12)


def test_penalty_is_exact_integral(knots):
    x, y = knots
    t = np.linspace(0.0, 1.0, 200_001)
    d2 = (evaluate_warps(x, y, t) - t) ** 2
    numeric = (d2[:, 1:] + d2[:, :-1]).sum(axis=1) * (t[1] - t[0]) / 2
    np.testing.assert_allclose(warp_penalty(x, y), numeric, rtol=1e-6)


def test_identity_knots():
    x, y = identity_knots(4, n_knots=2)
    t = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(evaluate_warps(x, y, t), np.tile(t, (4, 1)))
    assert x.shape == (4, 4)


def test_random_knots_respect_constraints():
    rng = np.random.default_rng(3)
    x, y = random_knots(500, 2, max_lag=0.1, max_stretch=1.5, rng=rng)
    assert x.shape == y.shape == (500, 4)
    slopes = np.diff(y, axis=1) / np.diff(x, axis=1)
    assert np.all(slopes >= 1 / 1.5 - 1e-12)
    assert np.all(slopes <= 1.5 + 1e-12)
    assert np.all(np.abs(y - x) <= 0.1 + 1e-12)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        evaluate_warps(np.zeros((2, 3)), np.zeros((2, 4)), [0.5])
    with pytest.raises(ValueError):
        evaluate_warps(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((3, 5)))
