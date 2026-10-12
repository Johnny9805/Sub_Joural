import numpy as np
import scipy.sparse as sp

from dsaudit.operators import (
    assignment_operator,
    block_average_operator,
    block_projector,
    multivariable_operator,
)


def test_block_average_matches_reshape_mean():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((12, 8))
    A = block_average_operator((12, 8), (3, 4))
    expected = x.reshape(4, 3, 2, 4).mean(axis=(1, 3)).ravel()
    np.testing.assert_allclose(A @ x.ravel(), expected)
    np.testing.assert_allclose(A @ np.ones(96), np.ones(8))


def test_assignment_operator_reduces_to_block_average_on_aligned_grid():
    n, f = 16, 4
    hr = (np.arange(n) + 0.5) / n  # cell centres on [0, 1]
    lr = (np.arange(n // f) + 0.5) / (n // f)
    lat, lon = np.meshgrid(hr, hr, indexing="ij")
    A, kept = assignment_operator(lat, lon, lr, lr)
    np.testing.assert_array_equal(kept, np.arange((n // f) ** 2))
    np.testing.assert_allclose(A.toarray(), block_average_operator((n, n), f).toarray())


def test_assignment_operator_descending_coarse_axis_and_empty_cells():
    lat = np.array([[10.1, 10.1], [10.4, 10.4]])
    lon = np.array([[120.0, 120.1], [120.0, 120.1]])
    lr_lat = np.array([11.0, 10.75, 10.5, 10.25, 10.0])  # ERA5-style descending
    lr_lon = np.array([119.75, 120.0, 120.25])
    A, kept = assignment_operator(lat, lon, lr_lat, lr_lon)
    assert A.shape == (2, 4)
    np.testing.assert_allclose(A.sum(axis=1).A.ravel(), 1.0)
    # 10.1 -> row 4 (10.0), 10.4 -> row 2 (10.5); lon 120.0/120.1 -> col 1
    np.testing.assert_array_equal(np.sort(kept), [2 * 3 + 1, 4 * 3 + 1])


def test_multivariable_operator_skips_variables_without_coarse_input():
    A1 = block_average_operator((4, 4), 2)
    M = multivariable_operator([A1, None, A1], 16)
    assert M.shape == (8, 48)
    assert sp.csr_matrix(M[:, 16:32]).nnz == 0


def test_block_projector_is_orthogonal_projector():
    A = block_average_operator((8, 8), 4)
    P = block_projector(A).toarray()
    np.testing.assert_allclose(P @ P, P, atol=1e-12)
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    np.testing.assert_allclose(np.trace(P), 4)
