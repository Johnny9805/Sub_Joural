"""Ensemble calibration diagnostics, overall and per provenance subspace.

Conventions: ``ens`` has the ensemble on axis 0, shape (K, ...); ``obs`` has
the remaining shape. Within a subspace the diagnostics use whitened
coordinates, where a correctly specified Gaussian posterior has unit
spread-skill ratio in every direction.
"""

from __future__ import annotations

import numpy as np

from .decomposition import SUBSPACES, ProvenanceBasis


def rank_histogram(ens: np.ndarray, obs: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Counts of the observation's rank among K members (K + 1 bins).

    Ties are broken uniformly at random.
    """
    rng = np.random.default_rng() if rng is None else rng
    ens = np.asarray(ens)
    K = ens.shape[0]
    e = ens.reshape(K, -1)
    o = np.asarray(obs).reshape(-1)
    below = np.sum(e < o, axis=0)
    ties = np.sum(e == o, axis=0)
    ranks = below + rng.integers(0, ties + 1)
    return np.bincount(ranks, minlength=K + 1)


def rank_flatness(counts: np.ndarray) -> float:
    """Chi-square statistic of a rank histogram against the uniform distribution,
    divided by its degrees of freedom (about 1 for a flat histogram)."""
    counts = np.asarray(counts, dtype=float)
    expected = counts.sum() / counts.size
    return float(np.sum((counts - expected) ** 2 / expected) / (counts.size - 1))


def spread_skill(ens: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    """Spread, RMSE of the ensemble mean, and the finite-ensemble corrected ratio.

    ratio = sqrt((K + 1) / K) * spread / rmse, which is 1 in expectation for a
    calibrated ensemble (Fortin et al. 2014, J. Hydrometeor.).
    """
    ens = np.asarray(ens)
    K = ens.shape[0]
    mean = ens.mean(axis=0)
    spread = np.sqrt(np.mean(ens.var(axis=0, ddof=1)))
    rmse = np.sqrt(np.mean((mean - obs) ** 2))
    return {"spread": float(spread), "rmse": float(rmse), "ratio": float(np.sqrt((K + 1) / K) * spread / rmse)}


def subspace_spread_skill(basis: ProvenanceBasis, ens: np.ndarray, obs: np.ndarray) -> dict[str, dict[str, float]]:
    """Spread-skill per subspace, in whitened coordinates.

    ens : (K, ..., n) members; obs : (..., n) truth. Leading dimensions after
    the ensemble axis are cases and are pooled. Spread and RMSE are per
    subspace dimension, so the three subspaces are directly comparable.
    """
    ens = np.asarray(ens)
    K = ens.shape[0]
    z_ens = basis.prior.whiten(ens)
    z_obs = basis.prior.whiten(obs)
    z_mean = z_ens.mean(axis=0)
    err = basis.split_whitened(z_mean - z_obs)
    dev = basis.split_whitened(z_ens - z_mean)
    dims = basis.dims
    out = {}
    for s in SUBSPACES:
        if dims[s] == 0:
            continue
        n_cases = int(np.prod(z_obs.shape[:-1]))
        mse = np.sum(err[s] ** 2) / (n_cases * dims[s])
        var = np.sum(dev[s] ** 2) / ((K - 1) * n_cases * dims[s])
        out[s] = {
            "spread": float(np.sqrt(var)),
            "rmse": float(np.sqrt(mse)),
            "ratio": float(np.sqrt((K + 1) / K) * np.sqrt(var / mse)),
        }
    return out


def subspace_rank_histograms(
    basis: ProvenanceBasis,
    ens: np.ndarray,
    obs: np.ndarray,
    n_prior_directions: int = 64,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Rank histograms of the truth's projection onto subspace directions.

    Data and coupled subspaces use their basis vectors; the prior-fill subspace
    (no explicit basis) uses ``n_prior_directions`` random unit directions
    projected into it.
    """
    rng = np.random.default_rng() if rng is None else rng
    z_ens = basis.prior.whiten(np.asarray(ens))
    z_obs = basis.prior.whiten(np.asarray(obs))
    dirs = {"data": basis.V_data, "coupled": basis.V_coupled}
    if basis.dims["prior"] > 0:
        R = basis.project_prior(rng.standard_normal((n_prior_directions, basis.prior.size)))
        dirs["prior"] = (R / np.linalg.norm(R, axis=1, keepdims=True)).T
    return {s: rank_histogram(z_ens @ D, z_obs @ D, rng) for s, D in dirs.items() if D.shape[1] > 0}
