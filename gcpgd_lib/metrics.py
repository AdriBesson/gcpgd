r"""Performance metrics and statistical utilities."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def wilson_ci(k, n, z=1.96):
    r"""Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


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
