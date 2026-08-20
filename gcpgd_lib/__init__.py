r"""GCPGD primitives shared across the reproduction scripts.

Submodules:
  toeplitz      Toeplitz weights/build/adjoint, rank projection, Cadzow denoiser
  signal        FRI location sampling, Fourier coefficients, measurement operator
  algorithm     Gamma-gradient GCPGD iteration and Gamma-norm factory
  certificates  Nontangentiality certificate, restricted/global mu, Lemma 4 bound
  recovery      Annihilating-filter location recovery + circular match error
  stats         Wilson binomial interval
"""

from .toeplitz import (toeplitz_weights, build_toeplitz, toeplitz_adjoint_read,
                       project_toeplitz, project_rank, cadzow_denoiser, robust_svd)
from .signal import sample_locations, fri_fourier, measurement_operator
from .algorithm import gcpgd, gnorm_factory, cadzow_denoise_pyoneer, run_cpgd, run_genfri
from .certificates import (check_nontangentiality, mu_gamma_full,
                           mu_restricted, sigmaK_lemma4)
from .recovery import recover_locations, circular_match_error, average_match_error
from .stats import wilson_ci

__all__ = [
    "toeplitz_weights", "build_toeplitz", "toeplitz_adjoint_read",
    "project_toeplitz", "project_rank", "cadzow_denoiser", "robust_svd",
    "sample_locations", "fri_fourier", "measurement_operator",
    "gcpgd", "gnorm_factory", "cadzow_denoise_pyoneer", "run_cpgd", "run_genfri",
    "check_nontangentiality", "mu_gamma_full", "mu_restricted", "sigmaK_lemma4",
    "recover_locations", "circular_match_error", "average_match_error",
    "wilson_ci",
]
