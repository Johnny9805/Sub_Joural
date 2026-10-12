import numpy as np
import pytest

from dsaudit.calibration import rank_flatness, rank_histogram, spread_skill
from dsaudit.spectra import band_fractions, radial_energy_spectrum


def test_rank_histogram_flat_for_exchangeable_members():
    rng = np.random.default_rng(0)
    ens = rng.standard_normal((9, 20_000))
    obs = rng.standard_normal(20_000)
    counts = rank_histogram(ens, obs, rng)
    assert counts.sum() == 20_000 and counts.size == 10
    assert rank_flatness(counts) < 3.0


def test_rank_histogram_detects_underdispersion():
    rng = np.random.default_rng(1)
    counts = rank_histogram(0.3 * rng.standard_normal((9, 20_000)), rng.standard_normal(20_000), rng)
    assert counts[0] > 3 * counts[5] and counts[-1] > 3 * counts[5]


def test_spread_skill_ratio_near_one_when_calibrated():
    rng = np.random.default_rng(2)
    mu = rng.standard_normal(50_000)
    ens = mu + rng.standard_normal((5, 50_000))
    obs = mu + rng.standard_normal(50_000)
    assert spread_skill(ens, obs)["ratio"] == pytest.approx(1.0, abs=0.02)


def test_radial_spectrum_parseval():
    f = np.random.default_rng(3).standard_normal((2, 32, 32))
    _, E = radial_energy_spectrum(f)
    np.testing.assert_allclose(E.sum(axis=-1), np.mean(f**2, axis=(-2, -1)))


def test_band_fractions_split_large_and_small_scales():
    n = 32
    yy, xx = np.mgrid[0:n, 0:n]
    large = np.cos(2 * np.pi * xx / n).ravel()  # wavelength 32 cells
    small = np.cos(2 * np.pi * 8 * yy / n).ravel()  # wavelength 4 cells
    fr = band_fractions({"large": large, "small": small}, (n, n), [np.inf, 16, 2])
    np.testing.assert_allclose(fr["large"], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(fr["small"], [0.0, 1.0], atol=1e-12)
