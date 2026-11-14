import numpy as np
import pytest

import trialwarp as tw


def test_consistency_of_identical_and_independent_trials():
    rng = np.random.default_rng(0)
    trial = rng.standard_normal((50, 3))
    assert tw.trial_consistency(np.repeat(trial[None], 8, axis=0)) == pytest.approx(1.0)
    noise = rng.standard_normal((200, 50, 3))
    assert abs(tw.trial_consistency(noise)) < 0.05


def test_consistency_ignores_nan_and_mask():
    rng = np.random.default_rng(1)
    X = np.repeat(rng.standard_normal((1, 30, 2)), 6, axis=0)
    X[0, :5] = np.nan
    assert tw.trial_consistency(X) == pytest.approx(1.0)
    noisy = X + 0.0
    noisy[2, 10:] += 100.0
    mask = np.ones(X.shape, dtype=bool)
    mask[2, 10:] = False
    assert tw.trial_consistency(noisy, mask=mask) == pytest.approx(1.0)
    assert tw.trial_consistency(X[:, :, 0]) == pytest.approx(1.0)  # 2D input


def test_heldout_score_detects_real_warps():
    data = tw.make_warped_trials(
        n_trials=50, n_timepoints=60, n_features=8, noise=0.4, random_state=0
    )
    model = tw.PiecewiseWarping(n_knots=1, random_state=0)
    score = tw.heldout_feature_score(model, data.data, n_folds=4, random_state=0, n_iter=6)
    assert score.raw.shape == score.aligned.shape == (4,)
    assert np.all(score.improvement > 0.05)
    assert "improvement" in str(score)


def test_heldout_score_finds_nothing_in_unwarped_data():
    data = tw.make_warped_trials(
        n_trials=50, n_timepoints=60, n_features=8, noise=0.4, max_lag=0.0, random_state=1
    )
    model = tw.PiecewiseWarping(n_knots=1, random_state=0)
    score = tw.heldout_feature_score(model, data.data, n_folds=4, random_state=0, n_iter=6)
    assert score.improvement.mean() < 0.01


def test_heldout_score_needs_features():
    with pytest.raises(ValueError, match="two features"):
        tw.heldout_feature_score(tw.ShiftWarping(), np.zeros((5, 10, 1)))
