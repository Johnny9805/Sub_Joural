"""Noise-weighted Gauss-Newton kernel and the three provenance subspaces.

Notation (all in prior-whitened coordinates z = L^{-1}(x - mean), B = L L^T):

    G = R^{-1/2} J L                 whitened Jacobian of a measurement y = J x + e
    G^T G = V diag(s2) V^T           Gauss-Newton (Fisher) kernel
    a_i = s2_i / (1 + s2_i)          averaging-kernel (resolution) eigenvalues
    DOFS = sum_i a_i                 degrees of freedom for signal

a_i is the fraction of direction v_i's posterior mean that is set by the
measurement rather than the prior; a_i >= 0.5 means the measurement beats the
prior there (s2_i >= 1).

Three subspaces (definition v0, see docs/method_notes.md):

    data-trusted   S_d = span{v_i : a_i >= tau} for J = linear coarse-graining A
    coupled        S_c = same construction for the coupling Jacobian J_c,
                         restricted to the orthogonal complement of S_d
    prior-fill     S_p = complement of S_d + S_c

The subspaces are orthogonal in whitened space, so whitened energies add up
exactly. Physical (unwhitened) energies of the components do not, because the
split is B^{-1}-orthogonal; the remainder is reported as a cross term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .priors import GaussianPrior

SUBSPACES = ("data", "coupled", "prior")


@dataclass
class GNKernel:
    """Eigen-decomposition of G^T G, keeping the non-zero part."""

    V: np.ndarray  # (n, r), orthonormal columns
    s2: np.ndarray  # (r,), descending

    @property
    def resolution(self) -> np.ndarray:
        return self.s2 / (1.0 + self.s2)

    @property
    def dofs(self) -> float:
        return float(self.resolution.sum())

    def subspace(self, tau: float) -> np.ndarray:
        return self.V[:, self.resolution >= tau]


def whitened_jacobian(J, noise_std, prior: GaussianPrior) -> np.ndarray:
    """G = R^{-1/2} J L for diagonal R; J may be sparse or dense (m, n)."""
    noise_std = np.broadcast_to(np.asarray(noise_std, dtype=float), (J.shape[0],))
    JL = J @ prior.L
    return np.asarray(JL) / noise_std[:, None]


def gn_kernel(G: np.ndarray, rtol: float = 1e-12) -> GNKernel:
    """Non-zero eigenpairs of G^T G, via the smaller of G G^T and G^T G."""
    m, n = G.shape
    if m <= n:
        s2, U = np.linalg.eigh(G @ G.T)
        s2, U = s2[::-1], U[:, ::-1]
        keep = s2 > rtol * max(s2.max(initial=0.0), np.finfo(float).tiny)
        s2, U = s2[keep], U[:, keep]
        V = (G.T @ U) / np.sqrt(s2)
    else:
        s2, V = np.linalg.eigh(G.T @ G)
        s2, V = s2[::-1], V[:, ::-1]
        keep = s2 > rtol * max(s2.max(initial=0.0), np.finfo(float).tiny)
        s2, V = s2[keep], V[:, keep]
    return GNKernel(V=V, s2=s2)


@dataclass
class ProvenanceBasis:
    prior: GaussianPrior
    V_data: np.ndarray
    V_coupled: np.ndarray
    tau: float
    kernel_data: GNKernel
    kernel_coupled: GNKernel | None = None
    dofs_total: float = field(default=np.nan)

    @property
    def dims(self) -> dict[str, int]:
        n, kd, kc = self.prior.size, self.V_data.shape[1], self.V_coupled.shape[1]
        return {"data": kd, "coupled": kc, "prior": n - kd - kc}

    @property
    def dofs_data(self) -> float:
        return self.kernel_data.dofs

    @property
    def dofs_converted(self) -> float:
        """DOFS gained by adding the coupling measurement (threshold-free H3 measure)."""
        return self.dofs_total - self.dofs_data

    def coefficients(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Coordinates of whitened vectors z (..., n) in the data and coupled bases."""
        return {"data": z @ self.V_data, "coupled": z @ self.V_coupled}

    def split_whitened(self, z: np.ndarray) -> dict[str, np.ndarray]:
        c = self.coefficients(z)
        zd = c["data"] @ self.V_data.T
        zc = c["coupled"] @ self.V_coupled.T
        return {"data": zd, "coupled": zc, "prior": z - zd - zc}

    def project_prior(self, z: np.ndarray) -> np.ndarray:
        """Orthogonal projection of whitened vectors onto the prior-fill subspace."""
        return self.split_whitened(z)["prior"]

    def split(self, x: np.ndarray, anomaly: bool = False) -> dict[str, np.ndarray]:
        """Physical components of x (or of an anomaly) that sum to x - mean."""
        parts = self.split_whitened(self.prior.whiten(x, anomaly=anomaly))
        return {k: self.prior.color(v) for k, v in parts.items()}

    def energy(self, x: np.ndarray, anomaly: bool = False) -> dict[str, dict[str, np.ndarray]]:
        """Energy fractions of x - mean (or of an anomaly) in each subspace.

        Returns {"whitened": {...}, "physical": {...}}; whitened fractions sum
        to 1, physical fractions sum to 1 together with the "cross" entry.
        """
        z = self.prior.whiten(x, anomaly=anomaly)
        zparts = self.split_whitened(z)
        ez = np.sum(z**2, axis=-1)
        white = {k: np.sum(v**2, axis=-1) / ez for k, v in zparts.items()}
        d = x if anomaly else x - self.prior.mean
        ed = np.sum(d**2, axis=-1)
        phys = {k: np.sum(self.prior.color(v) ** 2, axis=-1) / ed for k, v in zparts.items()}
        phys["cross"] = 1.0 - sum(phys.values())
        return {"whitened": white, "physical": phys}


def build_provenance_basis(
    prior: GaussianPrior,
    J_data,
    noise_data,
    J_coupled=None,
    noise_coupled=None,
    tau: float = 0.5,
) -> ProvenanceBasis:
    """Construct S_d, S_c and S_p from the linear and coupling Jacobians.

    Parameters
    ----------
    prior : HR prior N(mean, L L^T).
    J_data : (m, n) linear coarse-graining operator (sparse or dense).
    noise_data : coarse-input error std, scalar or (m,).
    J_coupled : optional (m_c, n) coupling Jacobian, e.g. rows built from model
        vector-Jacobian products, or terrain/lapse-rate functionals.
    noise_coupled : error std of the coupling rows.
    tau : averaging-kernel threshold; 0.5 means signal-to-noise >= 1.
    """
    Gd = whitened_jacobian(J_data, noise_data, prior)
    Kd = gn_kernel(Gd)
    Vd = Kd.subspace(tau)
    if J_coupled is None:
        return ProvenanceBasis(
            prior=prior,
            V_data=Vd,
            V_coupled=np.zeros((prior.size, 0)),
            tau=tau,
            kernel_data=Kd,
            dofs_total=Kd.dofs,
        )
    if noise_coupled is None:
        raise ValueError("noise_coupled is required with J_coupled")
    Gc = whitened_jacobian(J_coupled, noise_coupled, prior)
    Gc_perp = Gc - (Gc @ Vd) @ Vd.T
    Kc = gn_kernel(Gc_perp)
    Vc = Kc.subspace(tau)
    dofs_total = gn_kernel(np.vstack([Gd, Gc])).dofs
    return ProvenanceBasis(
        prior=prior,
        V_data=Vd,
        V_coupled=Vc,
        tau=tau,
        kernel_data=Kd,
        kernel_coupled=Kc,
        dofs_total=dofs_total,
    )


def empirical_averaging_kernel(prior: GaussianPrior, ens: np.ndarray) -> GNKernel:
    """Jacobian-free estimate of the averaging kernel from ensembles over many cases.

    In a linear-Gaussian problem with truth ~ N(mean, B), the law of total
    covariance gives, in whitened coordinates,

        Cov_cases(posterior mean) = I - (I + G^T G)^{-1} = V diag(a) V^T,

    i.e. the averaging kernel. With K members the ensemble mean carries extra
    variance P/K, which is removed using the pooled within-case covariance.

    ens : (K, cases, n) physical members, cases drawn from the prior climate.
    Returns a GNKernel whose ``resolution`` holds the estimated a_i, clipped to
    [0, 1). For a real model this is the model's own claim of how much is set
    by its inputs; compare it with the truth-based calibration diagnostics.
    Finite samples bias the null-space eigenvalues upward by roughly
    2 sqrt(n / cases) / K.
    """
    z = prior.whiten(np.asarray(ens))
    K, N, _ = z.shape
    zm = z.mean(axis=0)
    C = np.cov(zm, rowvar=False)
    dev = (z - zm).reshape(K * N, -1)
    W = dev.T @ dev / ((K - 1) * N)
    a, V = np.linalg.eigh(C - W / K)
    a, V = a[::-1], V[:, ::-1]
    a = np.clip(a, 0.0, 1.0 - 1e-9)
    return GNKernel(V=V, s2=a / (1.0 - a))


def materialize_rows(vjp: Callable[[np.ndarray], np.ndarray], m: int, n: int) -> np.ndarray:
    """Dense (m, n) Jacobian from a vector-Jacobian product, one row per call.

    Use with an autodiff VJP of the downscaler (e.g. torch.func.vjp) to build
    J_coupled for a small sub-domain.
    """
    J = np.empty((m, n))
    e = np.zeros(m)
    for i in range(m):
        e[i] = 1.0
        J[i] = vjp(e)
        e[i] = 0.0
    return J
