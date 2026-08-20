r"""Location recovery (annihilating filter) and circular matching error."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .toeplitz import build_toeplitz, robust_svd as svd


def recover_locations(x, N, P, K):
    r"""Estimate K Dirac locations from Fourier vector x via annihilating filter.

    The annihilating filter is the right singular vector of T_P(x) for the
    smallest singular value; its polynomial roots are exp(-j 2 pi t_k).
    """
    A = build_toeplitz(x, N, P)
    _, _, Vh = svd(A, full_matrices=False)
    h = Vh.conj()[-1]
    roots = np.roots(h)
    order = np.argsort(np.abs(np.abs(roots) - 1.0))
    roots = roots[order[:K]]
    t = np.mod(-np.angle(roots) / (2.0 * np.pi), 1.0)
    return np.sort(t)


def _match_distances(t_true, t_est):
    r"""Matched circular location errors under the optimal assignment."""
    D = np.abs(t_true[:, None] - t_est[None, :])
    D = np.minimum(D, 1.0 - D)
    ri, ci = linear_sum_assignment(D)
    return D[ri, ci]


def circular_match_error(t_true, t_est):
    r"""Max matched circular location error under the optimal assignment."""
    return _match_distances(t_true, t_est).max()


def average_match_error(t_true, t_est):
    r"""Mean matched circular location error under the optimal assignment."""
    return float(_match_distances(t_true, t_est).mean())
