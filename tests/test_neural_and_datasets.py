import numpy as np
import pytest

import trialwarp as tw


def test_bin_spikes_counts_and_centers():
    trials = [0, 0, 1, 1, 1, 0]
    times = [0.05, 0.95, 0.5, 0.51, 1.0, 1.5]  # 1.0 lands in the last bin, 1.5 is dropped
    units = [0, 1, 1, 1, 0, 0]
    counts, centers = tw.bin_spikes(trials, times, units, n_trials=2, n_units=2, n_bins=10)
    assert counts.shape == (2, 10, 2)
    assert counts.sum() == 5
    assert counts[0, 0, 0] == 1 and counts[0, 9, 1] == 1
    assert counts[1, 5, 1] == 2 and counts[1, 9, 0] == 1
    np.testing.assert_allclose(centers, np.arange(10) / 10 + 0.05)


def test_bin_spikes_validates_indices():
    with pytest.raises(IndexError):
        tw.bin_spikes([3], [0.5], [0], n_trials=2, n_units=1, n_bins=4)
    with pytest.raises(ValueError):
        tw.bin_spikes([0, 1], [0.5], [0], n_trials=2, n_units=1, n_bins=4)


def test_smooth_trials_keeps_constants_and_nan_padding():
    X = np.full((3, 40, 2), 5.0)
    np.testing.assert_allclose(tw.smooth_trials(X, 3.0), X)
    X[1, 30:] = np.nan
    out = tw.smooth_trials(X, 3.0)
    assert np.isnan(out[1, 30:]).all()
    np.testing.assert_allclose(out[1, :30], 5.0)
    assert tw.smooth_trials(X[:, :, 0], 2.0).shape == (3, 40)


def test_warped_trials_dataset():
    data = tw.make_warped_trials(
        n_trials=7, n_timepoints=30, n_features=4, n_knots=2, random_state=0
    )
    assert data.data.shape == data.clean.shape == (7, 30, 4)
    assert data.template.shape == (30, 4)
    assert data.x_knots.shape == (7, 4)
    assert data.warping_functions().shape == (7, 30)
    null = tw.make_warped_trials(n_trials=5, n_timepoints=30, max_lag=0.0, random_state=0)
    np.testing.assert_allclose(null.x_knots, null.y_knots)
    tight = tw.make_warped_trials(n_trials=5, n_timepoints=30, max_lag=1e-4, random_state=0)
    assert tight.x_knots.shape == (5, 3)  # very small lags still give every trial a warp


def test_spike_dataset():
    spikes = tw.make_spike_trials(n_trials=10, n_neurons=6, duration=2.0, random_state=0)
    assert spikes.trials.shape == spikes.times.shape == spikes.neurons.shape
    assert spikes.times.min() >= 0.0 and spikes.times.max() < 2.0
    assert set(np.unique(spikes.trials)) <= set(range(10))
    assert spikes.event_times.shape == (10,)
    assert np.all((spikes.event_times > 0) & (spikes.event_times < 2.0))
    assert spikes.peak_times.shape == (6,)
