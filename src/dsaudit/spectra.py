"""Radially averaged spectra and scale-band energy fractions."""

from __future__ import annotations

import numpy as np


def radial_energy_spectrum(field: np.ndarray, dx: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Energy per radial wavenumber shell of a periodic 2-D field.

    field : (..., ny, nx). Returns (k, E) with k in cycles per unit length
    (shell centres) and E of shape (..., n_shells); E sums over shells to
    mean(field**2) (Parseval), so it is an energy density per shell, not per
    unit wavenumber.
    """
    f = np.asarray(field, dtype=float)
    ny, nx = f.shape[-2:]
    if ny != nx:
        raise ValueError("radial spectra assume a square domain")
    F = np.fft.fft2(f)
    power = np.abs(F) ** 2 / (ny * nx) ** 2
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    shell = np.rint(kr).astype(int).ravel()
    n_shells = shell.max() + 1
    onehot = np.zeros((ny * nx, n_shells))
    onehot[np.arange(ny * nx), shell] = 1.0
    E = power.reshape(*power.shape[:-2], ny * nx) @ onehot
    k = np.arange(n_shells) / (nx * dx)
    return k, E


def band_fractions(
    components: dict[str, np.ndarray],
    shape: tuple[int, int],
    edges_cells: list[float],
) -> dict[str, np.ndarray]:
    """Share of each component's energy in wavelength bands.

    components : name -> flattened fields (..., ny*nx) that sum to the total.
    edges_cells : wavelength band edges in grid cells, descending
        (e.g. [inf, 64, 16, 4, 2]). Returns name -> (..., n_bands) fractions of
        the total energy in each band, so that summing over names gives 1 in
        every band (up to cross terms, which are returned as "cross").
    """
    ny, nx = shape
    spectra = {}
    for name, v in components.items():
        _, E = radial_energy_spectrum(np.asarray(v).reshape(*np.shape(v)[:-1], ny, nx))
        spectra[name] = E
    total_field = sum(np.asarray(v) for v in components.values())
    _, E_tot = radial_energy_spectrum(total_field.reshape(*total_field.shape[:-1], ny, nx))
    n_shells = E_tot.shape[-1]
    shell_wavelength = np.full(n_shells, np.inf)
    shell_wavelength[1:] = nx / np.arange(1, n_shells)
    out = {name: [] for name in spectra}
    out["cross"] = []
    for hi, lo in zip(edges_cells[:-1], edges_cells[1:]):
        sel = (shell_wavelength <= hi) & (shell_wavelength > lo)
        tot = E_tot[..., sel].sum(axis=-1)
        acc = 0.0
        for name, E in spectra.items():
            frac = E[..., sel].sum(axis=-1) / tot
            out[name].append(frac)
            acc = acc + frac
        out["cross"].append(1.0 - acc)
    return {k: np.stack(v, axis=-1) for k, v in out.items()}
