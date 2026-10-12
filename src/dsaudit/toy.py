"""Synthetic terrain / lapse-rate test problem with an exact Gaussian posterior.

Used to validate the pipeline before real CorrDiff output exists: in a
linear-Gaussian world the exact posterior sampler is calibrated by
construction, so every subspace must show a spread-skill ratio near 1, and
any deviation measured on CorrDiff can be attributed to the model.

Model (anomalies of 2 m temperature on an n x n grid, block size f):

    x = x_grf + sum_b Gamma_b h'_b        x_grf ~ stationary power-law field
    y_b = mean of x over block b + e       coarse input (ERA5-like)
    c_b = <h'_b, x> / |h'_b|^2 + e_c       local lapse-rate estimate (coupling)

where h'_b is the terrain anomaly inside block b and Gamma_b ~ N(0, s_G^2)
is the block's lapse-rate anomaly. c_b mimics what the 850/925 hPa
temperatures in the input tell a downscaler about the sub-grid distribution
of temperature along the terrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .decomposition import GNKernel, gn_kernel, whitened_jacobian
from .priors import GaussianPrior, power_law_spectrum, sample_stationary, stationary_covariance


def ridge_terrain(n: int, peak_km: float = 3.0, roughness: float = 0.1, seed: int = 0) -> np.ndarray:
    """A NNE-SSW Gaussian ridge (Central-Range-like) with small-scale roughness, in km."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    ang = np.deg2rad(20.0)
    across = xx * np.cos(ang) - yy * np.sin(ang)
    along = xx * np.sin(ang) + yy * np.cos(ang)
    ridge = peak_km * np.exp(-0.5 * (across / 0.09) ** 2) * np.exp(-0.5 * (along / 0.3) ** 2)
    rough = sample_stationary(power_law_spectrum((n, n), slope=3.0, k0=4.0), rng)[0]
    rough = roughness * peak_km * rough / np.abs(rough).max()
    return np.maximum(ridge + rough * (ridge / peak_km), 0.0)


def block_terrain_patterns(h: np.ndarray, factor: int, min_std_km: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Terrain anomaly patterns h'_b for blocks with enough relief.

    Returns (U, blocks): U is (n_kept, n) with row b = h'_b on its block and 0
    elsewhere; blocks lists the kept block indices (row-major over the coarse grid).
    """
    n = h.shape[0]
    nb = n // factor
    U, blocks = [], []
    for b in range(nb * nb):
        by, bx = divmod(b, nb)
        sl = (slice(by * factor, (by + 1) * factor), slice(bx * factor, (bx + 1) * factor))
        hb = h[sl] - h[sl].mean()
        if hb.std() < min_std_km:
            continue
        row = np.zeros_like(h)
        row[sl] = hb
        U.append(row.ravel())
        blocks.append(b)
    return np.array(U).reshape(-1, n * n), np.array(blocks, dtype=int)


def lapse_rate_operator(U: np.ndarray) -> sp.csr_matrix:
    """Coupling rows c_b = <h'_b, x> / |h'_b|^2 (regression slope of x on terrain in block b)."""
    return sp.csr_matrix(U / np.sum(U**2, axis=1, keepdims=True))


def toy_covariance(
    n: int,
    U: np.ndarray,
    grf_variance: float = 1.0,
    slope: float = 3.0,
    k0: float = 2.0,
    lapse_std: float = 2.0,
    nugget: float = 0.01,
) -> np.ndarray:
    """B = B_grf + lapse_std^2 U^T U + nugget I (dense)."""
    B = stationary_covariance(power_law_spectrum((n, n), slope=slope, k0=k0, variance=grf_variance))
    B += lapse_std**2 * (U.T @ U)
    B[np.diag_indices_from(B)] += nugget
    return B


@dataclass
class LinearGaussianPosterior:
    """Exact posterior of a linear-Gaussian problem, sampled in whitened space."""

    prior: GaussianPrior
    kernel: GNKernel
    z_mean: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        return self.prior.mean + self.prior.color(self.z_mean)

    def whitened_deviations(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draws from N(0, (I + G^T G)^{-1}) in whitened space, shape (size, n)."""
        V, s2 = self.kernel.V, self.kernel.s2
        w = rng.standard_normal((size, self.prior.size))
        shrink = 1.0 - 1.0 / np.sqrt(1.0 + s2)
        return w - ((w @ V) * shrink) @ V.T

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return self.mean + self.prior.color(self.whitened_deviations(rng, size))


class LinearGaussianModel:
    """Measurement operators [(J, noise_std), ...] with the GN kernel precomputed,
    so that posteriors for many observation vectors are cheap."""

    def __init__(self, prior: GaussianPrior, operators: list[tuple]):
        self.prior = prior
        self.operators = [(J, np.broadcast_to(np.asarray(s, dtype=float), (J.shape[0],))) for J, s in operators]
        self.G = np.vstack([whitened_jacobian(J, s, prior) for J, s in self.operators])
        self.kernel = gn_kernel(self.G)

    def posterior(self, ys: list[np.ndarray]) -> LinearGaussianPosterior:
        innov = np.concatenate([(y - J @ self.prior.mean) / s for (J, s), y in zip(self.operators, ys)])
        K = self.kernel
        z_mean = K.V @ ((K.V.T @ (self.G.T @ innov)) / (1.0 + K.s2))
        return LinearGaussianPosterior(prior=self.prior, kernel=K, z_mean=z_mean)


def exact_posterior(prior: GaussianPrior, measurements: list[tuple]) -> LinearGaussianPosterior:
    """Posterior given measurements [(J, noise_std, y), ...] with y = J x + e."""
    model = LinearGaussianModel(prior, [(J, s) for J, s, _ in measurements])
    return model.posterior([y for _, _, y in measurements])
