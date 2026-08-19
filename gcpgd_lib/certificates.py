r"""Geometric certificates and theoretical bounds.

check_nontangentiality : Definition 2 principal-angle certificate for T_P(x)
mu_gamma_full          : global weighted injectivity of G
mu_restricted          : restricted injectivity on the Gamma-orthonormalized tangent
sigmaK_lemma4          : Lemma 4 lower bound on sigma_K
"""

import numpy as np

from .toeplitz import build_toeplitz, toeplitz_weights, robust_svd as svd


def check_nontangentiality(x, N, P, K, tol_unit=1e-6):
    r"""Principal-angle certificate for the nontangentiality of T_P(x).

    Computes the principal angles between the Toeplitz subspace T_P and the
    tangent space of the rank-K manifold at M* = T_P(x). At a nontangential
    model point exactly 2K of the cosines equal 1 (the shared tangent
    directions of the exponential model manifold), and the next cosine is the
    angle constant c < 1 whose square-root-free value enters the rate.
    """
    # Build the Toeplitz matrix and get singular vectors for the rank-K manifold
    m1, m2 = N - P, P + 1
    A = build_toeplitz(x, N, P)
    U, s, Vh = svd(A, full_matrices=False)
    Uk = U[:, :K]
    Vk = Vh.conj().T[:, :K]

    # Q1: Construct an orthonormal basis for the Toeplitz subspace T_P.
    # Each column corresponds to one of the N diagonals, scaled by 1/sqrt(w_i)
    # to ensure orthonormality under the standard Frobenius inner product.
    w = toeplitz_weights(N, P)
    rr = np.arange(m1)[:, None]
    cc = np.arange(m2)[None, :]
    idx = rr - cc + P
    Q1 = np.zeros((m1 * m2, N), dtype=complex)
    for i in range(N):
        B = (idx == i).astype(float) / np.sqrt(w[i])
        Q1[:, i] = B.reshape(-1, order='F')

    # Mt: Build a redundant spanning set for the tangent space of the rank-K manifold.
    # Stack flattened matrices of the form Uk[:, i] @ e_c^T and e_r @ Vk[:, j]^H.
    cols = []
    I1 = np.eye(m1)
    I2 = np.eye(m2)
    for i in range(K):
        for c2 in range(m2):
            cols.append(np.outer(Uk[:, i], I2[c2]).reshape(-1, order='F'))
    for j in range(K):
        for r2 in range(m1):
            cols.append(
                np.outer(I1[r2], Vk[:, j].conj()).reshape(-1, order='F'))
    Mt = np.array(cols).T

    # Q2: Orthonormalize Mt using SVD to get a true orthonormal basis for the tangent space.
    # Columns associated with singular values below the numerical threshold are dropped.
    Q2, s2, _ = svd(Mt, full_matrices=False)
    rank2 = int((s2 > s2[0] * 1e-10).sum())
    Q2 = Q2[:, :rank2]

    # Compute principal cosines as singular values of the cross-correlation matrix Q1^H @ Q2
    sv = svd(Q1.conj().T @ Q2, compute_uv=False)
    n_unit = int((sv > 1.0 - tol_unit).sum())
    c = float(sv[n_unit]) if n_unit < len(sv) else 1.0
    return dict(n_unit=n_unit,
                expected_unit=2 * K,
                c=c,
                alpha=float(np.arccos(min(c, 1.0))),
                sigma_K=float(s[K - 1]),
                dims=(N, rank2, m1 * m2))


def mu_gamma_full(G, w):
    r"""Global weighted injectivity: sigma_min(G Gamma^{-1/2}) (full column rank)."""
    return float(svd(G * (1.0 / np.sqrt(w))[None, :], compute_uv=False)[-1])


def mu_restricted(G, w, t, a, M):
    r"""Restricted injectivity: sigma_min of G on the Gamma-orthonormalized
    tangent of the exponential model at x*(t, a) -- the local quantity that
    governs the contraction q and the stability constant 2/mu of Theorem 2.
    The global sigma_min(G Gamma^{-1/2}) is degenerate for irregular Fourier
    sampling, which is precisely why the restricted notion is used.
    """
    # Exponents corresponding to the Fourier series coefficients
    m = np.arange(-M, M + 1)
    z = np.exp(-2j * np.pi * t)
    
    # B: Construct the tangent matrix of size N x 2K containing partial derivatives
    # with respect to amplitudes (z_k^m) and locations (a_k * m * z_k^{m-1}).
    cols = [z_k**m for z_k in z]
    cols += [a_k * m * z_k**(m - 1) for a_k, z_k in zip(a, z)]
    B = np.array(cols).T
    
    # Orthonormalize the tangent basis under the Gamma-weighted vector norm.
    # Since the Gamma norm is ||v||_Gamma = ||v * sqrt(w)||_2, we scale B by
    # sqrt(w) and compute the standard QR decomposition to obtain Q.
    sw = np.sqrt(w)
    Q, _ = np.linalg.qr(B * sw[:, None])
    
    # Any vector in the tangent space has the Gamma-orthonormal representation Q / sqrt(w).
    # We find the minimum singular value of G restricted to this orthonormalized subspace.
    return float(svd(G @ (Q / sw[:, None]), compute_uv=False)[-1])


def sigmaK_lemma4(amin, N, P, Delta):
    r"""Lemma 4 lower bound on sigma_K; NaN where the validity condition
    min(N-P, P+1) > 1 + 1/Delta fails (bound vacuous below Rayleigh limit).
    """
    m1, m2 = N - P, P + 1
    if min(m1, m2) <= 1 + 1.0 / Delta:
        return float('nan')
    return amin * np.sqrt(m1 - 1.0 / Delta - 1) * np.sqrt(m2 - 1.0 / Delta - 1)
