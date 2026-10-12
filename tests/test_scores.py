import numpy as np
import pytest

from dsaudit.scores import (
    correntropy,
    correntropy_induced_metric,
    crps_ensemble,
    crps_gaussian,
    kernel_score_ensemble,
    kernel_width,
)


def _crps_bruteforce(x, y, fair):
    K = len(x)
    pair = np.abs(x[:, None] - x[None, :]).sum()
    return np.mean(np.abs(x - y)) - 0.5 * pair / (K * (K - 1) if fair else K * K)


@pytest.mark.parametrize("fair", [True, False])
def test_crps_ensemble_matches_bruteforce(fair):
    rng = np.random.default_rng(0)
    ens = rng.standard_normal((7, 20))
    obs = rng.standard_normal(20)
    got = crps_ensemble(ens, obs, fair=fair)
    want = [_crps_bruteforce(ens[:, j], obs[j], fair) for j in range(20)]
    np.testing.assert_allclose(got, want)


def test_crps_large_ensemble_matches_gaussian_closed_form():
    rng = np.random.default_rng(1)
    ens = 0.3 + 1.7 * rng.standard_normal((200_000, 1))
    obs = np.array([1.1])
    assert crps_ensemble(ens, obs)[0] == pytest.approx(crps_gaussian(0.3, 1.7, obs)[0], rel=5e-3)


def test_kernel_score_limits():
    obs = np.array([0.4])
    assert kernel_score_ensemble(np.full((5, 1), 0.4), obs, 1.0, fair=False)[0] == pytest.approx(0.0)
    far = kernel_score_ensemble(np.full((5, 1), 1e6), obs, 1.0, fair=False)[0]
    assert far == pytest.approx(1.0)
    # wide kernel -> (ensemble-mean error)^2 / (2 sigma^2)
    ens = np.array([[0.0], [1.0], [2.0]])
    sigma = 1e3
    ks = kernel_score_ensemble(ens, np.array([0.5]), sigma, fair=False)[0]
    assert ks == pytest.approx(0.5**2 / (2 * sigma**2), rel=1e-4)


def test_kernel_score_prefers_the_true_distribution():
    rng = np.random.default_rng(2)
    obs = rng.standard_normal(4000)
    scores = {
        s: kernel_score_ensemble(s * rng.standard_normal((40, obs.size)), obs, 1.0).mean()
        for s in (0.5, 1.0, 2.0)
    }
    assert scores[1.0] < scores[0.5] and scores[1.0] < scores[2.0]


def test_kernel_score_is_bounded_under_outliers_while_crps_is_not():
    rng = np.random.default_rng(3)
    ens = rng.standard_normal((20, 1))
    clean, outlier = np.array([0.0]), np.array([50.0])
    assert kernel_score_ensemble(ens, outlier, 1.0)[0] <= 1.0 + 1e-12
    assert crps_ensemble(ens, outlier)[0] > 40


def test_correntropy_and_cim():
    e = np.zeros(10)
    assert correntropy(e, 1.0) == pytest.approx(1.0)
    assert correntropy_induced_metric(e, 1.0) == pytest.approx(0.0)
    assert 0 < correntropy_induced_metric(np.array([3.0]), 1.0) < 1


def test_kernel_width_estimates_scale_robustly():
    rng = np.random.default_rng(4)
    r = 2.0 * rng.standard_normal(50_000)
    r[:500] = 1e4  # 1% gross errors
    assert kernel_width(r, "mad") == pytest.approx(2.0, rel=0.05)
    assert kernel_width(r, "silverman") > 0


@pytest.mark.parametrize("fair", [True, False])
def test_kernel_score_matches_bruteforce(fair):
    rng = np.random.default_rng(5)
    ens, obs, sigma = rng.standard_normal((6, 5)), rng.standard_normal(5), 0.7
    K = ens.shape[0]
    pairs = [(i, k) for i in range(K) for k in range(K) if i != k or not fair]
    k_xx = np.mean([np.exp(-0.5 * ((ens[i] - ens[k]) / sigma) ** 2) for i, k in pairs], axis=0)
    k_xy = np.mean(np.exp(-0.5 * ((ens - obs) / sigma) ** 2), axis=0)
    np.testing.assert_allclose(kernel_score_ensemble(ens, obs, sigma, fair=fair), 0.5 * k_xx - k_xy + 0.5)
