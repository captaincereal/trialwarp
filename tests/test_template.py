import numpy as np
import pytest

from trialwarp._template import (
    fit_template,
    interp_index,
    roughness_bands,
    warp_template,
)


def dense_interp_matrix(i0, w, n_bins):
    """W such that W @ template interpolates the template at every sample."""
    W = np.zeros((i0.size, n_bins))
    rows = np.arange(i0.size)
    W[rows, i0.ravel()] += 1.0 - w.ravel()
    W[rows, i0.ravel() + 1] += w.ravel()
    return W


@pytest.mark.parametrize("n_bins", [3, 4, 5, 12])
def test_roughness_bands_match_dense(n_bins):
    D = np.diff(np.eye(n_bins), n=2, axis=0)
    G = D.T @ D
    d0, d1, d2 = roughness_bands(n_bins)
    np.testing.assert_allclose(d0, np.diag(G))
    np.testing.assert_allclose(d1, np.diag(G, 1))
    np.testing.assert_allclose(d2, np.diag(G, 2))


def test_interpolation_matches_np_interp():
    rng = np.random.default_rng(0)
    template = rng.standard_normal((20, 3))
    pos = rng.uniform(-0.2, 1.2, (4, 30))
    i0, w = interp_index(pos, 20)
    out = warp_template(template, np.diff(template, axis=0), i0, w)
    grid = np.linspace(0.0, 1.0, 20)
    for n in range(3):
        expected = np.interp(np.clip(pos, 0, 1), grid, template[:, n])
        np.testing.assert_allclose(out[..., n], expected, atol=1e-12)


def solve_dense(X, mask, i0, w, n_bins, roughness, ridge):
    """Reference solution of the same penalized least-squares problem."""
    n_trials, n_time, n_features = X.shape
    W = dense_interp_matrix(i0, w, n_bins)
    D = np.diff(np.eye(n_bins), n=2, axis=0)
    flat = X.reshape(-1, n_features)
    if mask is None:
        mask = np.ones((n_trials, n_time))
    out = np.empty((n_bins, n_features))
    full = mask if mask.ndim == 3 else np.repeat(mask[..., None], n_features, axis=2)
    scale = full.sum() / (n_bins * n_features)
    for n in range(n_features):
        m = full[..., n].ravel()
        A = W.T @ (m[:, None] * W) + scale * (roughness * D.T @ D + ridge * np.eye(n_bins))
        out[:, n] = np.linalg.solve(A, W.T @ (m * flat[:, n]))
    return out


@pytest.mark.parametrize("mask_kind", ["none", "samples", "entries"])
def test_fit_template_matches_dense_solution(mask_kind):
    rng = np.random.default_rng(1)
    n_trials, n_time, n_features, n_bins = 6, 15, 3, 11
    X = rng.standard_normal((n_trials, n_time, n_features))
    pos = np.clip(np.linspace(0, 1, n_time) + rng.normal(0, 0.05, (n_trials, n_time)), 0, 1)
    i0, w = interp_index(np.sort(pos, axis=1), n_bins)
    mask = None
    if mask_kind == "samples":
        mask = (rng.random((n_trials, n_time)) > 0.2).astype(float)
    elif mask_kind == "entries":
        mask = (rng.random(X.shape) > 0.2).astype(float)
    if mask is not None:
        X = X * (mask if mask.ndim == 3 else mask[..., None])
    got = fit_template(X, mask, i0, w, n_bins, roughness=2.0, ridge=0.1)
    want = solve_dense(X, mask, i0, w, n_bins, roughness=2.0, ridge=0.1)
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)


def test_unpenalized_template_is_average_for_identity_warps():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((8, 10, 2))
    i0, w = interp_index(np.tile(np.linspace(0, 1, 10), (8, 1)), 10)
    got = fit_template(X, None, i0, w, 10, roughness=0.0, ridge=0.0)
    np.testing.assert_allclose(got, X.mean(axis=0), atol=1e-8)
