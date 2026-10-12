"""Provenance and calibration audit of generative weather downscaling.

Modules
-------
operators      linear coarse-graining forward operators (the "measurement")
priors         stationary Gaussian priors and matrix square roots
decomposition  noise-weighted Gauss-Newton kernel and the three subspaces
               (data-trusted / coupling-trusted / prior-fill)
calibration    rank histograms and spread-skill ratios, per subspace
scores         MSE, CRPS and correntropy (Gaussian-kernel) scores
spectra        radially averaged power spectra and scale-band energies
toy            synthetic terrain/lapse-rate test problem with an exact posterior
"""

__version__ = "0.1.0"
