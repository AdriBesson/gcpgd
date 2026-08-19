r"""GCPGD (Algorithm 1) with the Gamma-gradient step, and the Gamma-norm factory."""

import numpy as np

from .toeplitz import (cadzow_denoiser, build_toeplitz, project_rank,
                       toeplitz_adjoint_read)


def gnorm_factory(w):
    r"""Return u -> ||u||_Gamma with Gamma = diag(w)."""
    return lambda u: float(np.sqrt(np.real(np.vdot(u, w * u))))


def gcpgd(y,
          G,
          x0,
          N,
          P,
          K,
          w,
          n_cadzow=5,
          alpha=0.5,
          max_iter=2000,
          tol=1e-6,
          return_iter=False):
    r"""Run GCPGD with the Gamma-gradient step; return (x, converged_flag).

    v = x - 2 tau Gamma^{-1} G^H (G x - y);  z = H_n(v);  x = alpha z + (1-a) v.
    tau = 1 / (2 ||G||_Gamma^2), ||G||_Gamma = sigma_max(G Gamma^{-1/2}).
    Convergence flag: relative Gamma-norm step below tol.
    """
    Ginv_sqrt = 1.0 / np.sqrt(w)
    G_scaled = G * Ginv_sqrt[None, :]
    Gnorm_gamma = np.linalg.norm(G_scaled, 2)
    tau = 1.0 / (2.0 * Gnorm_gamma**2)
    Gh = G.conj().T
    gnorm = gnorm_factory(w)

    x = x0.copy()
    converged = False
    total_iter = 0
    for _ in range(max_iter):
        total_iter += 1
        grad = Gh @ (G @ x - y)
        v = x - 2.0 * tau * (grad / w)
        z = cadzow_denoiser(v, N, P, K, n_cadzow, w)
        x_new = alpha * z + (1.0 - alpha) * v
        if gnorm(x_new - x) <= tol * max(gnorm(x), 1e-12):
            x = x_new
            converged = True
            break
        x = x_new
    if return_iter:
        return x, converged, total_iter
    return x, converged


def proj_l2_ball(x, rho):
    r"""Project vector onto the l2 ball of radius rho."""
    norm = np.linalg.norm(x)
    if norm <= rho:
        return x
    return rho * x / norm


def cadzow_denoise_pyoneer(x, N, P, K, n_iter, w, rho=np.inf):
    r"""Run standard Cadzow denoising with an l2-ball constraint."""
    for _ in range(n_iter):
        x = proj_l2_ball(x, rho)
        A = build_toeplitz(x, N, P)
        A = project_rank(A, K)
        x = toeplitz_adjoint_read(A, N, P) / w
    return x


def run_cpgd(y, G, N, P, K, w, n_cadzow, max_iter=2000, tol=1e-7, rho=np.inf):
    r"""Reconstruct FRI coefficients using standard CPGD with projection constraint."""
    import time
    Gnorm = np.linalg.norm(G, 2)
    tau = 1.0 / (Gnorm ** 2)
    Gh = G.conj().T
    
    x = np.zeros(N, dtype=complex)
    total_iter = 0
    t_start = time.time()
    
    for it in range(max_iter):
        total_iter += 1
        x_old = x.copy()
        
        grad = Gh @ (G @ x - y)
        v = x - tau * grad
        
        x = cadzow_denoise_pyoneer(v, N, P, K, n_cadzow, w, rho)
        
        if np.linalg.norm(x_old) > 0:
            rel = np.linalg.norm(x - x_old) / np.linalg.norm(x_old)
        else:
            rel = np.inf
            
        if rel < tol:
            break
            
    elapsed = time.time() - t_start
    return x, total_iter, elapsed


def run_genfri(y, G, N, P, K, max_iter=50, nb_init=15, tol=1e-6, rcond=1e-4, seed=4):
    r"""Reconstruct FRI coefficients using Generalized FRI (GenFRI) algorithm."""
    import time
    import warnings
    import scipy.linalg as splin
    rng = np.random.default_rng(seed)
    
    Gh = G.conj().T
    G_gram = Gh @ G
    G_pinv = np.linalg.pinv(G, rcond=rcond)
    beta = G_pinv @ y
    
    T_beta = build_toeplitz(beta, N, P) / np.sqrt(P + 1)
    
    rhs4 = np.zeros(2 * N + 2, dtype=complex)
    rhs4[-1] = 1.0
    
    rhs5 = np.zeros(2 * N - P, dtype=complex)
    rhs5[:N] = Gh @ y
    
    best_b = None
    min_error = np.inf
    total_iter = 0
    t_start = time.time()
    
    for init in range(nb_init):
        c = rng.standard_normal(P + 1) + 1j * rng.standard_normal(P + 1)
        c0 = c.copy()
        
        row1 = np.zeros((P + 1, P + 1), dtype=complex)
        row2 = T_beta.transpose().conj()
        row3 = np.zeros((P + 1, N), dtype=complex)
        row4 = c0[:, None]
        mtx_top = np.concatenate((row1, row2, row3, row4), axis=1)
        
        mtx_bot = np.zeros((1, 2 * N + 2), dtype=complex)
        mtx_bot[0, :P + 1] = c0.conj()
        
        for it in range(max_iter):
            total_iter += 1
            
            col_c = np.concatenate(([c[-1]], np.zeros(N - P - 1)))
            row_c = np.concatenate((c[::-1], np.zeros(N - P - 1)))
            R_c = splin.toeplitz(col_c, row_c)
            
            mid2_1 = T_beta
            mid2_2 = np.zeros((N - P, N - P), dtype=complex)
            mid2_3 = -R_c
            mid2_4 = np.zeros((N - P, 1), dtype=complex)
            mtx_mid2 = np.concatenate((mid2_1, mid2_2, mid2_3, mid2_4), axis=1)
            
            mid3_1 = np.zeros((N, P + 1), dtype=complex)
            mid3_2 = -R_c.transpose().conj()
            mid3_3 = G_gram
            mid3_4 = np.zeros((N, 1), dtype=complex)
            mtx_mid3 = np.concatenate((mid3_1, mid3_2, mid3_3, mid3_4), axis=1)
            
            mtx_4 = np.concatenate((mtx_top, mtx_mid2, mtx_mid3, mtx_bot), axis=0)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = splin.solve(mtx_4, rhs4, check_finite=False, assume_a='her')
                
            c = sol[:P + 1]
            
        col_c = np.concatenate(([c[-1]], np.zeros(N - P - 1)))
        row_c = np.concatenate((c[::-1], np.zeros(N - P - 1)))
        R_c = splin.toeplitz(col_c, row_c)
        
        recon1 = np.concatenate((G_gram, R_c.transpose().conj()), axis=1)
        recon2 = np.concatenate((R_c, np.zeros((N - P, N - P), dtype=complex)), axis=1)
        mtx_5 = np.concatenate((recon1, recon2), axis=0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol_b = splin.solve(mtx_5, rhs5)[:N]
            
        err = np.linalg.norm(y - G @ sol_b)
        if err < min_error:
            min_error = err
            best_b = sol_b
            
    elapsed = time.time() - t_start
    return best_b, total_iter, elapsed
