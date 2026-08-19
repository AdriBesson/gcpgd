r"""FRI signal model: location sampling, Fourier coefficients, measurement operator."""

import numpy as np


def sample_locations(K, delta_sep, rng):
    r"""K locations in [0,1) with pairwise circular separation >= delta_sep.

    Direct construction on the simplex of slack; exact whenever K*delta_sep < 1.
    Rejection sampling is infeasible at the tight end of the grids.
    """
    if K * delta_sep >= 1.0:
        raise RuntimeError(
            f"infeasible: K*delta_sep = {K*delta_sep:.3f} >= 1 (K={K}, "
            f"delta_sep={delta_sep})")
    slack = 1.0 - K * delta_sep
    u = np.sort(rng.uniform(0, 1, size=K - 1))
    h = np.diff(np.concatenate([[0.0], u, [1.0]])) * slack
    gaps = delta_sep + h
    t0 = rng.uniform(0, 1)
    t = (t0 + np.concatenate([[0.0], np.cumsum(gaps[:-1])])) % 1.0
    return np.sort(t)


def fri_fourier(t, a, M):
    r"""x_hat_m = sum_k a_k exp(-j 2 pi m t_k), m=-M..M."""
    m = np.arange(-M, M + 1)
    return (a[None, :] * np.exp(-2j * np.pi * np.outer(m, t))).sum(axis=1)


def measurement_operator(N, L, M, rng):
    r"""G_{l,m} = exp(j 2 pi m theta_l), theta_l ~ U[0,1), m=-M..M. Shape L x N."""
    theta = np.sort(rng.uniform(0, 1, size=L))
    m = np.arange(-M, M + 1)
    return np.exp(2j * np.pi * np.outer(theta, m))
