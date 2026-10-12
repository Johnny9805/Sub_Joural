import numpy as np
import pytest

from dsaudit.calibration import subspace_spread_skill
from dsaudit.decomposition import build_provenance_basis, empirical_averaging_kernel, gn_kernel
from dsaudit.operators import block_average_operator, block_projector
from dsaudit.priors import gaussian_prior
from dsaudit.toy import (
    LinearGaussianModel,
    block_terrain_patterns,
    exact_posterior,
    lapse_rate_operator,
    ridge_terrain,
    toy_covariance,
)


@pytest.mark.parametrize("m,n", [(5, 12), (12, 5)])
def test_gn_kernel_matches_direct_eigendecomposition(m, n):
    G = np.random.default_rng(1).standard_normal((m, n))
    K = gn_kernel(G)
    H = G.T @ G
    np.testing.assert_allclose(K.V @ np.diag(K.s2) @ K.V.T, H, atol=1e-10)
    np.testing.assert_allclose(K.V.T @ K.V, np.eye(K.V.shape[1]), atol=1e-10)


def test_white_prior_data_subspace_is_block_means():
    n, f, sigma = 8, 4, 0.01
    A = block_average_operator((n, n), f)
    prior = gaussian_prior(np.eye(n * n))
    basis = build_provenance_basis(prior, A, sigma)
    # G G^T = A A^T / sigma^2 = I / (sigma f)^2 -> every block mean is resolved
    np.testing.assert_allclose(basis.kernel_data.s2, 1 / (sigma * f) ** 2)
    assert basis.dims == {"data": 4, "coupled": 0, "prior": 60}
    Pd = basis.V_data @ basis.V_data.T
    np.testing.assert_allclose(Pd, block_projector(A).toarray(), atol=1e-10)


def test_noisy_data_is_not_trusted():
    n, f = 8, 4
    A = block_average_operator((n, n), f)
    prior = gaussian_prior(np.eye(n * n))
    # block-mean prior variance is 1/f^2; noise std 1 makes s2 = 1/16 < 1
    basis = build_provenance_basis(prior, A, 1.0)
    assert basis.dims["data"] == 0
    assert basis.dofs_data == pytest.approx(4 * (1 / 16) / (1 + 1 / 16))


@pytest.fixture(scope="module")
def toy():
    n, f = 16, 4
    h = ridge_terrain(n, seed=3)
    U, _ = block_terrain_patterns(h, f, min_std_km=0.01)
    prior = gaussian_prior(toy_covariance(n, U))
    A = block_average_operator((n, n), f)
    Jc = lapse_rate_operator(U)
    basis = build_provenance_basis(prior, A, 0.2, Jc, 0.5)
    return dict(n=n, prior=prior, A=A, Jc=Jc, basis=basis)


def test_subspaces_are_orthogonal_and_energies_add_up(toy):
    basis = toy["basis"]
    V = np.hstack([basis.V_data, basis.V_coupled])
    np.testing.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-8)
    assert basis.dims["coupled"] > 0
    assert basis.dofs_total >= basis.dofs_data

    x = toy["prior"].sample(np.random.default_rng(0), 5)
    parts = basis.split(x)
    np.testing.assert_allclose(sum(parts.values()), x - toy["prior"].mean, atol=1e-8)
    e = basis.energy(x)
    np.testing.assert_allclose(sum(e["whitened"].values()), 1.0)
    np.testing.assert_allclose(sum(e["physical"].values()), 1.0)
    assert all(np.all(v >= -1e-12) for v in e["whitened"].values())


def test_exact_posterior_is_calibrated_in_every_subspace(toy):
    rng = np.random.default_rng(42)
    prior, A, Jc, basis = toy["prior"], toy["A"], toy["Jc"], toy["basis"]
    truths = prior.sample(rng, 300)
    ens = []
    for x in truths:
        y = A @ x + 0.2 * rng.standard_normal(A.shape[0])
        c = Jc @ x + 0.5 * rng.standard_normal(Jc.shape[0])
        post = exact_posterior(prior, [(A, 0.2, y), (Jc, 0.5, c)])
        ens.append(post.sample(rng, 8))
    ens = np.stack(ens, axis=1)  # (K, cases, n)
    ss = subspace_spread_skill(basis, ens, truths)
    for s, v in ss.items():
        assert v["ratio"] == pytest.approx(1.0, abs=0.1), (s, v)


def test_empirical_averaging_kernel_recovers_analytic_one(toy):
    rng = np.random.default_rng(7)
    prior, A, Jc = toy["prior"], toy["A"], toy["Jc"]
    model = LinearGaussianModel(prior, [(A, 0.2), (Jc, 0.5)])
    truths = prior.sample(rng, 3000)
    ens = []
    for x in truths:
        y = A @ x + 0.2 * rng.standard_normal(A.shape[0])
        c = Jc @ x + 0.5 * rng.standard_normal(Jc.shape[0])
        ens.append(model.posterior([y, c]).sample(rng, 16))
    est = empirical_averaging_kernel(prior, np.stack(ens, axis=1))
    a_true = model.kernel.resolution
    r = a_true.size
    np.testing.assert_allclose(est.resolution[:r], a_true, atol=0.12)
    assert est.resolution[r:].max() < 0.1
    # leading subspace agrees with the analytic one
    overlap = np.linalg.svd(est.V[:, :10].T @ model.kernel.V[:, :r], compute_uv=False)
    assert overlap.min() > 0.9
