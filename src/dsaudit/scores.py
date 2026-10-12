"""Point and ensemble scores, including bounded correntropy-based scores.

All scores here are negatively oriented (lower is better) and returned per
point unless stated otherwise; average them yourself over the verification set.

Correntropy and the Gaussian kernel score
-----------------------------------------
With the Gaussian kernel k(a, b) = exp(-(a - b)^2 / (2 sigma^2)), the
correntropy of an error sample e is V = mean k(e, 0), and the
correntropy-induced metric is CIM = sqrt(1 - V). For an ensemble forecast the
matching proper score is the kernel score (half the squared MMD between the
forecast and the observation),

    KS(F, y) = 1/2 E k(X, X') - E k(X, y) + 1/2,

which is strictly proper for the Gaussian kernel (Gneiting & Raftery 2007,
Sec. 5) and bounded in [0, 1]: one outlying station value can move it by at
most 1, whereas it moves the CRPS linearly and the MSE quadratically. For
sigma much larger than the errors it reduces to (ensemble-mean error)^2 /
(2 sigma^2), i.e. a scaled MSE.

The kernel width must be fixed once (per variable, from a reference sample)
and shared by all forecasts being compared; a width fitted to each forecast's
own errors makes the score improper.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def mse(forecast: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return (np.asarray(forecast) - obs) ** 2


def crps_ensemble(ens: np.ndarray, obs: np.ndarray, fair: bool = True) -> np.ndarray:
    """CRPS of a K-member ensemble (axis 0), via the sorted-member identity.

    fair=True uses the unbiased K(K-1) normalisation of E|X - X'|
    (Ferro 2014), so the score does not reward small ensembles.
    """
    x = np.sort(np.asarray(ens, dtype=float), axis=0)
    K = x.shape[0]
    term1 = np.mean(np.abs(x - obs), axis=0)
    w = (2 * np.arange(K) - K + 1).reshape((K,) + (1,) * (x.ndim - 1))
    pair_sum = 2.0 * np.sum(w * x, axis=0)  # sum_{i,j} |x_i - x_j|
    denom = K * (K - 1) if fair else K * K
    return term1 - 0.5 * pair_sum / denom


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma^2)."""
    z = (np.asarray(obs) - mu) / sigma
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + erf(z / np.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))


def gaussian_kernel(d: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (np.asarray(d) / sigma) ** 2)


def correntropy(errors: np.ndarray, sigma: float, axis=None) -> np.ndarray:
    """Sample correntropy mean k(e, 0); higher means closer agreement."""
    return np.mean(gaussian_kernel(errors, sigma), axis=axis)


def correntropy_induced_metric(errors: np.ndarray, sigma: float, axis=None) -> np.ndarray:
    return np.sqrt(1.0 - correntropy(errors, sigma, axis=axis))


def kernel_score_ensemble(ens: np.ndarray, obs: np.ndarray, sigma: float, fair: bool = True) -> np.ndarray:
    """Gaussian kernel score of a K-member ensemble (axis 0); see module docstring."""
    x = np.asarray(ens, dtype=float)
    K = x.shape[0]
    k_xy = np.mean(gaussian_kernel(x - obs, sigma), axis=0)
    k_xx = np.zeros(x.shape[1:])  # sum over ordered pairs i != j, without a (K, K, ...) array
    for i in range(K - 1):
        k_xx += 2.0 * gaussian_kernel(x[i + 1 :] - x[i], sigma).sum(axis=0)
    if fair:
        k_xx = k_xx / (K * (K - 1))
    else:
        k_xx = (k_xx + K) / (K * K)
    return 0.5 * k_xx - k_xy + 0.5


def kernel_width(reference: np.ndarray, method: str = "mad", scale: float = 1.0) -> float:
    """Data-adaptive Gaussian kernel width from a reference sample.

    reference : e.g. climatological station anomalies, or the errors of a
        fixed baseline (such as bilinear interpolation of the coarse input).
    method : "mad" -> 1.4826 * median absolute deviation (robust std);
             "silverman" -> 1.06 * min(std, IQR / 1.349) * n^(-1/5).
    scale : multiplier, for sensitivity studies over the width.
    """
    r = np.asarray(reference, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if method == "mad":
        width = 1.4826 * np.median(np.abs(r - np.median(r)))
    elif method == "silverman":
        q75, q25 = np.percentile(r, [75, 25])
        width = 1.06 * min(r.std(ddof=1), (q75 - q25) / 1.349) * r.size ** (-0.2)
    else:
        raise ValueError(f"unknown method {method!r}")
    if not width > 0:
        raise ValueError("reference sample has zero spread")
    return float(scale * width)
