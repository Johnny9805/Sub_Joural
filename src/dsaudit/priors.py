"""Gaussian priors for the HR field and their square roots.

The decomposition works in prior-whitened coordinates z = L^{-1} (x - mean),
with B = L L^T. This module provides a stationary (periodic) power-law prior
for synthetic tests and a generic symmetric square root for any dense SPD B,
e.g. a shrinkage estimate from HR training samples.

Dense matrices are fine for the week-3 target (a sub-domain of a few thousand
grid points). The full 448x448x4 CorrDiff state needs a matrix-free version.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def power_law_spectrum(
    shape: tuple[int, int],
    slope: float,
    k0: float = 1.0,
    variance: float = 1.0,
) -> np.ndarray:
    """Isotropic spectral density P(k) proportional to (k0^2 + |k|^2)^(-slope/2).

    Wavenumbers are in grid units (cycles per domain). P is normalised so that a
    field drawn with it has pointwise variance ``variance`` (numpy FFT convention:
    Var = mean(P)). The k = 0 mode is kept.
    """
    ny, nx = shape
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    P = (k0**2 + k2) ** (-slope / 2.0)
    return P * (variance / P.mean())


def sample_stationary(spectrum: np.ndarray, rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Draw periodic Gaussian random fields with spectral density ``spectrum``.

    Returns an array of shape (size, ny, nx).
    """
    w = rng.standard_normal((size, *spectrum.shape))
    return np.real(np.fft.ifft2(np.sqrt(spectrum) * np.fft.fft2(w)))


def stationary_covariance(spectrum: np.ndarray) -> np.ndarray:
    """Dense covariance matrix (n x n, n = ny*nx) of the periodic field."""
    ny, nx = spectrum.shape
    c = np.real(np.fft.ifft2(spectrum))
    iy, ix = np.divmod(np.arange(ny * nx), nx)
    dy = (iy[:, None] - iy[None, :]) % ny
    dx = (ix[:, None] - ix[None, :]) % nx
    return c[dy, dx]


@dataclass
class GaussianPrior:
    """N(mean, B) with symmetric square root L = B^{1/2} and its inverse."""

    mean: np.ndarray
    L: np.ndarray
    Linv: np.ndarray

    @property
    def size(self) -> int:
        return self.mean.size

    def whiten(self, x: np.ndarray, anomaly: bool = False) -> np.ndarray:
        """z = L^{-1}(x - mean); x may have leading batch dimensions (..., n).

        Set ``anomaly=True`` for differences (errors, ensemble spread), which
        are whitened without subtracting the mean.
        """
        d = x if anomaly else x - self.mean
        return d @ self.Linv.T

    def color(self, z: np.ndarray) -> np.ndarray:
        """Inverse of ``whiten`` for anomalies: L z."""
        return z @ self.L.T

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        return self.mean + self.color(rng.standard_normal((size, self.size)))


def gaussian_prior(B: np.ndarray, mean: np.ndarray | None = None, rel_floor: float = 1e-8) -> GaussianPrior:
    """Build a GaussianPrior from a dense symmetric PSD covariance.

    Eigenvalues below ``rel_floor * max eigenvalue`` are raised to that floor
    so that L is invertible (a red spectrum leaves the smallest scales nearly
    degenerate).
    """
    B = 0.5 * (B + B.T)
    lam, Q = np.linalg.eigh(B)
    lam = np.maximum(lam, rel_floor * lam.max())
    s = np.sqrt(lam)
    L = (Q * s) @ Q.T
    Linv = (Q / s) @ Q.T
    if mean is None:
        mean = np.zeros(B.shape[0])
    return GaussianPrior(mean=np.asarray(mean, dtype=float), L=L, Linv=Linv)
