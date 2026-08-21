r"""Location recovery (annihilating filter)."""

import numpy as np

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
