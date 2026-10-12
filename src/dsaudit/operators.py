"""Linear coarse-graining operators.

The downscaling "measurement" is the high-resolution (HR) field seen through a
coarse grid: y = A x + noise. Fields are flattened row-major (C order); for
several variables the state is the concatenation [x_1, x_2, ...].

Interpolating y back onto the HR grid (as CorrDiff does with ERA5) is an
injective map of y and adds no information, so the information content of the
input is fully described by A.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def block_average_operator(shape: tuple[int, int], factor: int | tuple[int, int]) -> sp.csr_matrix:
    """Block-mean operator for a regular grid.

    Parameters
    ----------
    shape : (ny, nx) of the HR grid; each dimension must be divisible by factor.
    factor : coarsening factor, scalar or (fy, fx).

    Returns
    -------
    A : sparse (ny//fy * nx//fx, ny*nx); each row averages one block.
    """
    ny, nx = shape
    fy, fx = (factor, factor) if np.isscalar(factor) else factor
    if ny % fy or nx % fx:
        raise ValueError(f"shape {shape} not divisible by factor {(fy, fx)}")
    iy, ix = np.divmod(np.arange(ny * nx), nx)
    rows = (iy // fy) * (nx // fx) + (ix // fx)
    cols = np.arange(ny * nx)
    vals = np.full(ny * nx, 1.0 / (fy * fx))
    return sp.csr_matrix((vals, (rows, cols)), shape=((ny // fy) * (nx // fx), ny * nx))


def assignment_operator(
    hr_lat: np.ndarray,
    hr_lon: np.ndarray,
    lr_lat: np.ndarray,
    lr_lon: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Area-mean operator from a curvilinear HR grid to a regular lat-lon grid.

    Each HR point is assigned to the nearest coarse cell centre (e.g. ERA5
    0.25 deg); row b of A averages the HR points falling in cell b. Coarse cells
    that receive no HR point are dropped.

    Parameters
    ----------
    hr_lat, hr_lon : HR coordinates, any shape (flattened in C order).
    lr_lat, lr_lon : 1-D regular coarse coordinates (ascending or descending).

    Returns
    -------
    A : sparse (n_cells_kept, n_hr)
    kept : flat indices (into the lr_lat x lr_lon grid) of the kept cells, in row order.
    """
    lat = np.asarray(hr_lat, dtype=float).ravel()
    lon = np.asarray(hr_lon, dtype=float).ravel()
    if lat.shape != lon.shape:
        raise ValueError("hr_lat and hr_lon must have the same shape")
    iy = _nearest_regular(lat, np.asarray(lr_lat, dtype=float))
    ix = _nearest_regular(lon, np.asarray(lr_lon, dtype=float))
    cell = iy * len(lr_lon) + ix
    kept, rows = np.unique(cell, return_inverse=True)
    counts = np.bincount(rows)
    A = sp.csr_matrix((1.0 / counts[rows], (rows, np.arange(lat.size))), shape=(kept.size, lat.size))
    return A, kept


def _nearest_regular(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    step = axis[1] - axis[0]
    if not np.allclose(np.diff(axis), step):
        raise ValueError("coarse axis must be regularly spaced")
    return np.clip(np.rint((values - axis[0]) / step).astype(int), 0, axis.size - 1)


def multivariable_operator(per_variable: list[sp.spmatrix | None], hr_size: int) -> sp.csr_matrix:
    """Stack per-variable operators into one acting on [x_1, ..., x_C].

    A ``None`` entry means the variable has no coarse counterpart in the input
    (e.g. CorrDiff-Taiwan radar reflectivity): it contributes columns but no rows.
    """
    blocks = []
    for c, op in enumerate(per_variable):
        if op is None:
            continue
        if op.shape[1] != hr_size:
            raise ValueError(f"operator {c} has {op.shape[1]} columns, expected {hr_size}")
        pad_left = sp.csr_matrix((op.shape[0], c * hr_size))
        pad_right = sp.csr_matrix((op.shape[0], (len(per_variable) - c - 1) * hr_size))
        blocks.append(sp.hstack([pad_left, op, pad_right]))
    if not blocks:
        raise ValueError("at least one variable must have an operator")
    return sp.vstack(blocks).tocsr()


def block_projector(A: sp.spmatrix) -> sp.csr_matrix:
    """Euclidean orthogonal projector onto the row space of A, P = A^T (A A^T)^{-1} A.

    For a block-mean operator, P x replaces each block by its mean, and
    (I - P) x is the sub-grid part that the coarse field cannot see.
    """
    AAt = (A @ A.T).tocsc()
    if (AAt - sp.diags(AAt.diagonal())).count_nonzero() == 0:
        inv = sp.diags(1.0 / AAt.diagonal())
        return (A.T @ inv @ A).tocsr()
    return sp.csr_matrix(A.T @ spla.spsolve(AAt, A.tocsc()))
