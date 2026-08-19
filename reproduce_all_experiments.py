#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
reproduce_all_experiments.py
============================

Unified driver for the experiments of the GCPGD paper. Each subcommand
produces one figure/CSV pair and validates a named claim:

  rate      Certificate -> rate: measured per-cycle rate of the inner
            alternating projections vs. the prediction c^2 from the
            nontangentiality certificate (Definition 2, Lemma 2, Thm 1(iii)).
  geometry  Ensemble statistics of the certificate: c and sigma_K vs the
            separation Delta; n_unit = 2K throughout (Definition 2 generic).
  phase     Success-rate phase diagram over (PSNR, Delta) on the collision
            ensemble (one pair merging, sigma_K collapsing), warm-started at
            L = 2N; overlays: Theorem-2 threshold from measured (sigma_K, mu)
            [mechanism, parallel], its 1-constant calibration [shape hugs],
            and the explicit Corollary-2 curve on its validity region.
  outer     Outer-loop validation: noiseless linear rate vs q-tilde, and
            noise-linearity of the limiting error (Theorem 2).

Common options: --fast (default) / --full, --seed, --outdir.
The basin-collapse experiment lives in reproduce_basin_scaling.py.
"""

import argparse
import csv
import os

import numpy as np
from numpy.linalg import pinv

from gcpgd_lib import (build_toeplitz, cadzow_denoiser, check_nontangentiality,
                       fri_fourier, gcpgd, gnorm_factory, measurement_operator,
                       mu_restricted, project_rank, project_toeplitz,
                       sample_locations, sigmaK_lemma4, toeplitz_weights,
                       wilson_ci, recover_locations, circular_match_error,
                       toeplitz_adjoint_read, cadzow_denoise_pyoneer, run_cpgd,
                       run_genfri, average_match_error, robust_svd as svd)


def measure_ap_rate(x, N, P, K, w, sK, rng, iters=600, pert=0.05):
    r"""Run inner alternating projections from a tube perturbation; fit rate."""
    Mstar = build_toeplitz(x, N, P)
    d = rng.standard_normal(
        Mstar.shape) + 1j * rng.standard_normal(Mstar.shape)
    A = project_toeplitz(Mstar + pert * sK * d / np.linalg.norm(d), N, P, w)
    traj = [A]
    for _ in range(iters):
        A = project_toeplitz(project_rank(A, K), N, P, w)
        traj.append(A)
    Minf = traj[-1]
    e = np.array([np.linalg.norm(T - Minf) for T in traj[:-1]])
    mask = (e > 1e-10) & (e < e[0] * 0.5)
    ks = np.arange(len(e))[mask]
    if len(ks) < 10:
        return float('nan')
    return float(np.exp(np.polyfit(ks, np.log(e[mask]), 1)[0]))


# --------------------------------------------------------------------------- #
#  rate
# --------------------------------------------------------------------------- #
def cmd_rate(cfg, rng=None):
    rows = []
    for _sd in cfg.seeds:
        rng = np.random.default_rng(_sd)
        for (K, M) in cfg.rate_KM:
            N, P = 2 * M + 1, M
            w = toeplitz_weights(N, P)
            for dsep in cfg.rate_deltas:
                for _ in range(cfg.rate_trials):
                    t = sample_locations(K, dsep, rng)
                    a = np.exp(1j * rng.uniform(0, 2 * np.pi, K))
                    x = fri_fourier(t, a, M)
                    cert = check_nontangentiality(x, N, P, K)
                    if cert['n_unit'] != 2 * K:
                        rows.append((K, M, dsep, cert['n_unit'], np.nan,
                                     np.nan, cert['sigma_K']))
                        continue
                    rate = measure_ap_rate(x, N, P, K, w, cert['sigma_K'], rng)
                    rows.append((K, M, dsep, cert['n_unit'], cert['c']**2,
                                 rate, cert['sigma_K']))
    _write_csv(
        cfg, 'rate',
        ['K', 'M', 'delta', 'n_unit', 'c2_pred', 'rate_meas', 'sigma_K'], rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'plots', 'custom_style.mplstyle'))
        pred = np.array([r[4] for r in rows])
        meas = np.array([r[5] for r in rows])
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        ax.plot([0, 1], [0, 1],
                '--',
                color='#555555',
                lw=1.2,
                label='measured = predicted')
        ax.scatter(pred, meas, color='#348ABD', s=20, alpha=0.8)
        ax.set_xlabel(r'predicted rate $c^2$ (certificate)')
        ax.set_ylabel(r'measured per-cycle rate')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10, loc='upper left')
        fig.tight_layout()
        fig.savefig(_p(cfg, 'rate.pdf'), bbox_inches='tight', pad_inches=0.01)
        print('figure ->', _p(cfg, 'rate.pdf'))
    except Exception as e:
        print('plot skipped:', e)


# --------------------------------------------------------------------------- #
#  geometry
# --------------------------------------------------------------------------- #
def cmd_geometry(cfg, rng=None):
    K, M = cfg.geo_K, cfg.geo_M
    N, P = 2 * M + 1, M
    rows = []
    for _sd in cfg.seeds:
        rng = np.random.default_rng(_sd)
        for dsep in cfg.geo_deltas:
            for _ in range(cfg.geo_trials):
                t = sample_locations(K, dsep, rng)
                # varied amplitudes so min|a_k| (which drives the Lemma 4 bound)
                # is a genuine per-instance quantity, not a constant
                a = np.exp(1j * rng.uniform(0, 2 * np.pi, K)) * rng.uniform(
                    0.6, 1.4, K)
                x = fri_fourier(t, a, M)
                cert = check_nontangentiality(x, N, P, K)
                amin = float(np.min(np.abs(a)))
                bnd = sigmaK_lemma4(amin, N, P, dsep)
                rows.append((dsep, cert['n_unit'], cert['c'], cert['sigma_K'],
                             amin, bnd))
    _write_csv(cfg, 'geometry',
               ['delta', 'n_unit', 'c', 'sigma_K', 'amin', 'sigmaK_bound'],
               rows)
    bad = sum(1 for r in rows if r[1] != 2 * K)
    print(f'n_unit == 2K in {len(rows) - bad}/{len(rows)} instances')
    valid = [(r[3], r[5]) for r in rows if not np.isnan(r[5])]
    held = sum(1 for sK, b in valid if sK >= b - 1e-9)
    if valid:
        print(f'Lemma 4 bound <= measured sigma_K in {held}/{len(valid)} '
              f'instances where it applies')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'plots', 'custom_style.mplstyle'))
        ds = np.array(sorted(set(r[0] for r in rows)))
        col = lambda i, d: [r[i] for r in rows if r[0] == d]
        med = lambda i, d: np.median(col(i, d))
        q1 = lambda i, d: np.percentile(col(i, d), 25)
        q3 = lambda i, d: np.percentile(col(i, d), 75)

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 4.0))

        # -- panel A: measured sigma_K (with IQR band) vs Lemma 4 bound --
        m_sig = np.array([med(3, d) for d in ds])
        lo = np.array([q1(3, d) for d in ds])
        hi = np.array([q3(3, d) for d in ds])
        axL.fill_between(ds,
                         lo,
                         hi,
                         color='#348ABD',
                         alpha=0.30,
                         label='measured IQR')
        axL.plot(ds,
                 m_sig,
                 'o-',
                 color='#348ABD',
                 ms=4,
                 label=r'measured median')
        with np.errstate(all="ignore"):
            b_sig = np.array([
                np.nan if np.all(np.isnan(col(5, d))) else np.nanmedian(
                    col(5, d)) for d in ds
            ])
        axL.plot(ds, b_sig, 'k--', lw=1.5, label=r'Lemma 4 bound')
        d_edge = 1.0 / (min(N - P, P + 1) - 1)
        axL.axvline(d_edge,
                    color='#E24A33',
                    ls=':',
                    lw=1.5,
                    label=r'validity edge')
        axL.set_xlabel(r'separation $\Delta$')
        axL.set_xscale('log')
        axL.set_ylabel(r'$\sigma_K$')
        axL.set_ylim(bottom=0)
        axL.legend(fontsize=9, loc='upper left')
        axL.set_title(r'conditioning vs.\ Lemma 4', fontsize=11)

        # -- panel B: collision sweep -- drive ONE pair together (gap -> 0)
        #    and show that sigma_K collapses while c stays bounded away from 1.
        gaps = np.geomspace(0.10, 5e-4, 12)
        c_coll, sk_coll = [], []
        anchors = 0.55
        rng_b = np.random.default_rng(12345)
        for g in gaps:
            cc, ss = [], []
            for _ in range(max(5, cfg.geo_trials // 4)):
                extra = list(anchors + 0.12 * np.arange(K - 2))
                t = np.array([0.08, 0.08 + g] + extra)
                aa = np.exp(1j * rng_b.uniform(0, 2 * np.pi, K))
                cert = check_nontangentiality(fri_fourier(t, aa, M), N, P, K)
                cc.append(cert['c'])
                ss.append(cert['sigma_K'])
            c_coll.append(np.median(cc))
            sk_coll.append(np.median(ss))
        axB = axR
        axB.semilogx(gaps, c_coll, 'o-', color='#E24A33', ms=4)
        axB.set_xlabel(r'colliding-pair gap')
        axB.invert_xaxis()
        axB.set_ylabel(r'median angle constant $c$', color='#E24A33')
        axB.set_ylim(0, 1)
        axB.tick_params(axis='y', labelcolor='#E24A33')
        axB.axhline(1.0, color='#E24A33', ls=':', lw=1.0, alpha=0.8)
        axB2 = axB.twinx()
        axB2.semilogy(gaps, sk_coll, 's-', color='#348ABD', ms=4)
        axB2.set_ylabel(r'median $\sigma_K$ (log)', color='#348ABD')
        axB2.tick_params(axis='y', labelcolor='#348ABD')
        axB.set_title(r'collision: $\sigma_K \to 0$ but $c \not\to 1$',
                      fontsize=11)

        fig.tight_layout()
        fig.savefig(_p(cfg, 'geometry.pdf'),
                    bbox_inches='tight',
                    pad_inches=0.01)
        print('figure ->', _p(cfg, 'geometry.pdf'))
    except Exception as e:
        print('plot skipped:', e)


# --------------------------------------------------------------------------- #
#  phase
# --------------------------------------------------------------------------- #
def cmd_phase(cfg, rng=None):
    r"""Phase diagram on the COLLISION ensemble: one Dirac pair at gap Delta
    (the binding separation), remaining nodes fixed far apart, so that sigma_K
    genuinely collapses as Delta -> 0 and the transition slopes.

    Overlays: (a) the Theorem-2 warm-start threshold computed from the
    MEASURED per-gap medians of (sigma_K, mu) with the raw constant c2 --
    tests the mechanism (parallelism across the transition); (b) its
    one-constant calibration (dotted) -- the shape hugs; (c) the fully
    explicit Corollary-2 curve via the Moitra bound, drawn only where valid,
    with the validity edge marked -- the certified region, showing that the
    closed-form sigma_K bound, not the mechanism, is the limiting factor.

    Requires L = 2N: at L = N the irregular-Fourier operator is globally
    degenerate and the least-squares warm start is ill-posed."""
    K, M = cfg.ph_K, cfg.ph_M
    N, P = 2 * M + 1, M
    L = 2 * N
    w = toeplitz_weights(N, P)
    gn = gnorm_factory(w)
    gaps = np.array(cfg.ph_gaps)
    psnrs = np.array(cfg.ph_psnrs, dtype=float)
    succ = np.zeros((len(psnrs), len(gaps)), dtype=int)
    mu_all = [[] for _ in gaps]
    sK_all = [[] for _ in gaps]
    others = 0.5 + 0.16 * np.arange(max(K - 2, 0))
    for _sd in cfg.seeds:
        rng = np.random.default_rng(_sd)
        for gi, g in enumerate(gaps):
            for _ in range(cfg.ph_trials):
                t0 = rng.uniform(0, 1)
                t = np.mod(np.concatenate([[t0, t0 + g], t0 + others]), 1.0)
                a = np.exp(1j * rng.uniform(0, 2 * np.pi, K))
                xstar = fri_fourier(t, a, M)
                sK = float(
                    svd(build_toeplitz(xstar, N, P), compute_uv=False)[K - 1])
                sK_all[gi].append(sK)
                G = measurement_operator(N, L, M, rng)
                mu_all[gi].append(mu_restricted(G, w, t, a, M))
                Gp = pinv(G, rcond=1e-6)
                for pi, ps in enumerate(psnrs):
                    sig = np.exp(-ps / 10.0)  # R = max|a_k| = 1
                    eps = sig * (rng.standard_normal(L) +
                                 1j * rng.standard_normal(L)) / np.sqrt(2)
                    y = G @ xstar + eps
                    xh, _ = gcpgd(y,
                                  G,
                                  Gp @ y,
                                  N,
                                  P,
                                  K,
                                  w,
                                  n_cadzow=cfg.ph_ncad,
                                  alpha=0.5,
                                  max_iter=cfg.ph_maxit,
                                  tol=1e-12)
                    if gn(xh - xstar) <= 0.1 * sK:
                        succ[pi, gi] += 1
            print(
                f'[seed={_sd} gap={g:.4f}] '
                f'med sigma_K={np.median(sK_all[gi]):.3f} '
                f'med mu={np.median(mu_all[gi]):.3f}',
                flush=True)
    n_tot = cfg.ph_trials * len(cfg.seeds)
    mu_med = np.array([np.median(v) for v in mu_all])
    sK_med = np.array([np.median(v) for v in sK_all])
    _phase_emit(cfg,
                gaps,
                psnrs,
                succ,
                n_tot,
                mu_med,
                sK_med,
                N,
                P,
                L,
                seeds_tag='|'.join(str(x) for x in cfg.seeds))


def _phase_emit(cfg,
                gaps,
                psnrs,
                succ,
                n_tot,
                mu_med,
                sK_med,
                N,
                P,
                L,
                seeds_tag=''):
    r"""Write phase.csv (pooled counts + Wilson CIs), phase_meta.csv (per-gap
    medians and config needed to rebuild overlays), and the figure."""
    m1, m2 = N - P, P + 1
    rate = succ / float(n_tot)
    rows = []
    for pi in range(len(psnrs)):
        for gi in range(len(gaps)):
            lo, hi = wilson_ci(int(succ[pi, gi]), n_tot)
            rows.append((float(psnrs[pi]), float(gaps[gi]), float(rate[pi,
                                                                       gi]),
                         round(lo, 4), round(hi, 4), n_tot))
    _write_csv(
        cfg, 'phase',
        ['psnr', 'gap', 'success', 'wilson_lo', 'wilson_hi', 'n_trials'], rows)
    meta = [(float(g), float(mu_med[gi]), float(sK_med[gi]), cfg.ph_K,
             cfg.ph_M, L, n_tot, seeds_tag) for gi, g in enumerate(gaps)]
    _write_csv(cfg, 'phase_meta',
               ['gap', 'mu_med', 'sK_med', 'K', 'M', 'L', 'n_trials', 'seeds'],
               meta)
    delta_tube = 0.05
    # sharpened prox constant (rho/2 exterior-sphere inequality):
    # c2 = (1-delta)/(2(2-delta)), tube rho <= 1-delta, r_delta=(1-d)/(2-d) sK
    c2 = (1 - delta_tube) / (2 * (2 - delta_tube))
    th_meas = 10 * np.log(np.sqrt(L) / (c2 * mu_med * sK_med))
    emp = np.full(len(gaps), np.nan)
    for gi in range(len(gaps)):
        idx = np.where(rate[:, gi] >= 0.5)[0]
        if len(idx):
            emp[gi] = psnrs[idx[0]]
    good = ~np.isnan(emp)
    b = float(np.median(emp[good] - th_meas[good])) if good.any() else 0.0
    th_fit = th_meas + b
    moitra = np.array([sigmaK_lemma4(1.0, N, P, g) for g in gaps])
    th_cor2 = 10 * np.log(np.sqrt(L) / (c2 * mu_med * moitra))
    d_edge = 1.0 / (min(m1, m2) - 1)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'plots', 'custom_style.mplstyle'))
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        xs = np.arange(len(gaps)) + 0.5
        im = ax.imshow(rate,
                       origin='lower',
                       aspect='auto',
                       extent=[0, len(gaps), psnrs[0], psnrs[-1]],
                       cmap='viridis',
                       vmin=0,
                       vmax=1)
        ax.plot(xs,
                th_meas,
                color='#E24A33',
                ls='-',
                lw=2.2,
                label=r'Thm 2 threshold (measured $\sigma_K,\mu$)')
        ax.plot(xs,
                th_fit,
                'w:',
                lw=1.8,
                label=f'shape, 1-const fit ({b:+.0f} dB)')
        ax.plot(xs,
                th_cor2,
                color='#348ABD',
                ls='--',
                lw=2.0,
                marker='s',
                ms=4.5,
                label='Cor. 2 (explicit, valid)')
        if gaps.min() < d_edge < gaps.max():
            xe = float(np.interp(d_edge, gaps, xs))
            ax.axvline(xe, color='#555555', ls=':', lw=1.2, alpha=0.8)
            ax.text(xe,
                    psnrs[0] + 2,
                    ' validity edge',
                    fontsize=8,
                    rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{g:.3f}' for g in gaps], fontsize=8)
        ax.set_xlabel(r'minimum separation $\Delta$ (one pair merging)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_ylim(psnrs[0], psnrs[-1])
        ax.set_title(
            r'exact recovery (err $\leq 0.1\sigma_K$) '
            r'vs. predicted thresholds',
            fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        fig.colorbar(im, label='success rate')
        fig.tight_layout()
        fig.savefig(_p(cfg, 'phase.pdf'), bbox_inches='tight', pad_inches=0.01)
        print('figure ->', _p(cfg, 'phase.pdf'))
    except Exception as e:
        print('plot skipped:', e)


def cmd_vanilla(cfg, rng=None):
    r"""Vanilla FRI denoising (G = Id): GCPGD(Id) versus one-shot Cadzow.

    Corollary 3 makes every constant closed-form here (mu_Gamma =
    (P+1)^{-1/2}, tau = 1/2). This experiment probes the metric-bias
    hypothesis: plain Cadzow returns (essentially) the nearest model point in
    the Gamma/Frobenius metric, while GCPGD(Id) balances an UNWEIGHTED
    l2-fidelity to y against Gamma-model-consistency -- the maximum-likelihood
    metric for i.i.d. coefficient noise. Reported: coefficient error in both
    metrics and matched location error, medians over the ensemble."""
    K, M = cfg.va_K, cfg.va_M
    N, P = 2 * M + 1, M
    w = toeplitz_weights(N, P)
    gn = gnorm_factory(w)
    G = np.eye(N, dtype=complex)
    rows = []
    for _sd in cfg.seeds:
        rng = np.random.default_rng(_sd)
        for ps in cfg.va_psnrs:
            for _ in range(cfg.va_trials):
                t = sample_locations(K, cfg.va_delta, rng)
                a = np.exp(1j * rng.uniform(0, 2 * np.pi, K))
                xs = fri_fourier(t, a, M)
                sig = np.exp(-ps / 10.0)
                eps = sig * (rng.standard_normal(N) +
                             1j * rng.standard_normal(N)) / np.sqrt(2)
                y = xs + eps
                # one-shot Cadzow, run to convergence
                x_cad = cadzow_denoiser(y, N, P, K, cfg.va_ncad_oneshot, w)
                # GCPGD with G = Id, warm start y
                x_gc, _ = gcpgd(y,
                                G,
                                y.copy(),
                                N,
                                P,
                                K,
                                w,
                                n_cadzow=cfg.va_ncad,
                                alpha=0.5,
                                max_iter=cfg.va_maxit,
                                tol=1e-10)
                for tag, xh in (('cadzow', x_cad), ('gcpgd', x_gc)):
                    terr = circular_match_error(t,
                                                recover_locations(xh, N, P, K))
                    rows.append((
                        ps,
                        tag,
                        float(np.linalg.norm(xh - xs)),  # l2
                        gn(xh - xs),  # Gamma
                        float(terr)))
    _write_csv(cfg, 'vanilla',
               ['psnr', 'method', 'err_l2', 'err_gamma', 'err_loc'], rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'plots', 'custom_style.mplstyle'))
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
        for ai, (col, ttl, ylog) in enumerate([(2, r'$\ell_2$ error', True),
                                               (3, r'$\Gamma$ error', True),
                                               (4, 'location error', True)]):
            ax = axes[ai]
            for tag, sty in (('cadzow', 'o-'), ('gcpgd', 's--')):
                med = [
                    np.median(
                        [r[col] for r in rows if r[0] == ps and r[1] == tag])
                    for ps in cfg.va_psnrs
                ]
                ax.plot(cfg.va_psnrs, med, sty, ms=5, label=tag)
            if ylog:
                ax.set_yscale('log')
            ax.set_xlabel('PSNR (dB)', fontsize=11)
            ax.set_title(ttl, fontsize=12)
        axes[0].legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(_p(cfg, 'vanilla.pdf'),
                    bbox_inches='tight',
                    pad_inches=0.01)
        print('figure ->', _p(cfg, 'vanilla.pdf'))
    except Exception as e:
        print('plot skipped:', e)
    # summary ratios
    for col, name in ((2, 'l2'), (3, 'Gamma'), (4, 'loc')):
        rat = []
        for ps in cfg.va_psnrs:
            c = np.median(
                [r[col] for r in rows if r[0] == ps and r[1] == 'cadzow'])
            g = np.median(
                [r[col] for r in rows if r[0] == ps and r[1] == 'gcpgd'])
            rat.append(g / c)
        print(f'{name:>6} error ratio gcpgd/cadzow by PSNR: ' +
              ' '.join(f'{x:.3f}' for x in rat))


def cmd_certificate(cfg, rng=None):
    r"""A posteriori drift certificate (Remark 2): run the Cadzow inner loop
    from the theorem's own radius r_delta, accumulate the Lemma-1 product
    1+eps_hat = prod (1-rho_k)^{-1/2}, check G_delta membership, and compare
    the certified bound with the measured drift (bound must dominate)."""
    rows = []
    delta = 0.05
    for _sd in cfg.seeds:
        rng = np.random.default_rng(_sd)
        for (K, M) in cfg.ce_KM:
            N, P = 2 * M + 1, M
            w = toeplitz_weights(N, P)
            for dsep in cfg.ce_deltas:
                for _ in range(cfg.ce_trials):
                    t = sample_locations(K, dsep, rng)
                    a = np.exp(1j * rng.uniform(0, 2 * np.pi, K))
                    x = fri_fourier(t, a, M)
                    Ms = build_toeplitz(x, N, P)
                    sK = float(svd(Ms, compute_uv=False)[K - 1])
                    cert = check_nontangentiality(x, N, P, K)
                    r = (1 - delta) / (2 - delta) * sK
                    d = rng.standard_normal(N) + 1j * rng.standard_normal(N)
                    D = project_toeplitz(build_toeplitz(d, N, P), N, P, w)
                    D /= np.linalg.norm(D)
                    Mk = Ms + r * D
                    d0 = np.linalg.norm(Mk - Ms)
                    prod, drift, in_tube = 1.0, 0.0, True
                    for _k in range(cfg.ce_ncyc):
                        sv = svd(Mk, compute_uv=False)
                        rho = float(sv[K] / sv[K - 1])
                        if rho > 1 - delta:
                            in_tube = False
                        prod *= (1 - min(rho, 1 - 1e-12))**-0.5
                        Mk = project_toeplitz(project_rank(Mk, K), N, P, w)
                        drift = max(drift, np.linalg.norm(Mk - Ms) / d0)
                    rows.append((K, M, dsep, round(cert['c'], 5), round(sK, 4),
                                 round(prod - 1, 5), round(drift,
                                                           5), int(in_tube)))
    _write_csv(
        cfg, 'certificate',
        ['K', 'M', 'delta', 'c', 'sigma_K', 'eps_hat', 'drift', 'in_tube'],
        rows)
    eh = np.array([r[5] for r in rows])
    dr = np.array([r[6] for r in rows])
    it = np.array([r[7] for r in rows])
    ok = np.mean(1 + eh >= dr)
    print(f'eps_hat: median={np.median(eh):.4f} max={eh.max():.4f}  '
          f'in-tube={it.mean():.2%}  certified bound >= drift: {ok:.2%}')
    thr = 'min(1,(1-q)/(2q)) threshold depends on G; eps_hat above is G-free'
    print(thr)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'plots', 'custom_style.mplstyle'))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 4.0))
        cs = [r[3] for r in rows]
        a1.scatter(cs, eh, color='#348ABD', s=20, alpha=0.8)
        a1.set_xlabel(r'angle constant $c$', fontsize=11)
        a1.set_ylabel(r'certified $\hat{\varepsilon}$', fontsize=11)
        a2.scatter(dr, 1 + eh, color='#348ABD', s=20, alpha=0.8)
        lim = [0, max(1.05, (1 + eh).max() * 1.05)]
        a2.plot(lim,
                lim,
                '--',
                color='#E24A33',
                lw=1.5,
                label=r'certified = drift')
        a2.set_xlabel('measured drift', fontsize=11)
        a2.set_ylabel(r'certified bound $1+\hat{\varepsilon}$', fontsize=11)
        a2.legend(fontsize=9, loc='upper left')
        fig.tight_layout()
        fig.savefig(_p(cfg, 'certificate.pdf'),
                    bbox_inches='tight',
                    pad_inches=0.01)
        print('figure ->', _p(cfg, 'certificate.pdf'))
    except Exception as e:
        print('plot skipped:', e)


def cmd_phase_aggregate(cfg, rng=None):
    r"""Pool per-seed phase runs (produced with distinct --seed / --outdir)
    at the COUNT level and re-emit the pooled CSV, meta, and figure.

    Usage: reproduce_all_experiments.py phase-aggregate \
               --inputs results/s0,results/s1,... --outdir results/pooled
    Each input dir must contain phase.csv and phase_meta.csv."""
    import glob as _glob
    dirs = []
    for pat in cfg.inputs.split(','):
        hits = sorted(_glob.glob(pat))
        dirs.extend(hits if hits else [pat])
    assert dirs, 'no input directories'
    succ = None
    n_tot = 0
    mu_cols, sK_cols = [], []
    gaps = psnrs = None
    K = M = L = None
    for d in dirs:
        with open(os.path.join(d, 'phase.csv')) as f:
            rd = list(csv.reader(f))
        hdr = rd[0]
        col = {h: i for i, h in enumerate(hdr)}
        data = [[float(x) for x in r[:5]] + [int(r[5])] for r in rd[1:]]
        g_here = sorted(set(r[col['gap']] for r in data))
        p_here = sorted(set(r[col['psnr']] for r in data))
        if gaps is None:
            gaps, psnrs = np.array(g_here), np.array(p_here)
            succ = np.zeros((len(psnrs), len(gaps)), dtype=int)
        assert list(gaps) == g_here and list(psnrs) == p_here, \
            f'{d}: grid mismatch'
        n_here = data[0][col['n_trials']]
        for r in data:
            pi = int(np.searchsorted(psnrs, r[col['psnr']]))
            gi = int(np.searchsorted(gaps, r[col['gap']]))
            succ[pi, gi] += int(round(r[col['success']] * n_here))
        n_tot += int(n_here)
        with open(os.path.join(d, 'phase_meta.csv')) as f:
            md = list(csv.reader(f))
        mcol = {h: i for i, h in enumerate(md[0])}
        mrows = md[1:]
        mu_cols.append([float(r[mcol['mu_med']]) for r in mrows])
        sK_cols.append([float(r[mcol['sK_med']]) for r in mrows])
        K = int(mrows[0][mcol['K']])
        M = int(mrows[0][mcol['M']])
        L = int(mrows[0][mcol['L']])
        print(f'pooled {d}: n={n_here}')
    N, P = 2 * M + 1, M
    cfg.ph_K, cfg.ph_M = K, M
    mu_med = np.median(np.array(mu_cols), axis=0)
    sK_med = np.median(np.array(sK_cols), axis=0)
    _phase_emit(cfg,
                gaps,
                psnrs,
                succ,
                n_tot,
                mu_med,
                sK_med,
                N,
                P,
                L,
                seeds_tag=f'aggregated:{len(dirs)}runs')


def cmd_outer(cfg, rng):
    K, M = cfg.ou_K, cfg.ou_M
    N, P = 2 * M + 1, M
    L = 2 * N  # oversampled: global mu is degenerate for irregular Fourier at L=N
    w = toeplitz_weights(N, P)
    gn = gnorm_factory(w)
    t = sample_locations(K, cfg.ou_delta, rng)
    a = np.exp(1j * rng.uniform(0, 2 * np.pi, K))
    xstar = fri_fourier(t, a, M)
    sK = float(svd(build_toeplitz(xstar, N, P), compute_uv=False)[K - 1])
    G = measurement_operator(N, L, M, rng)
    muG = mu_restricted(G, w, t, a, M)
    Gs = G * (1.0 / np.sqrt(w))[None, :]
    tau = 1.0 / (2.0 * np.linalg.norm(Gs, 2)**2)
    q = np.sqrt(max(0.0, 1 - 2 * tau * muG**2))
    qt = (1 + q) / 2
    # (a) noiseless linear rate
    y = G @ xstar
    d = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    x = xstar + 0.2 * sK * d / gn(d)
    errs = []
    Gh = G.conj().T
    for _ in range(cfg.ou_iters):
        errs.append(gn(x - xstar))
        v = x - 2 * tau * ((Gh @ (G @ x - y)) / w)
        z = cadzow_denoiser(v, N, P, K, cfg.ou_ncad, w)
        x = 0.5 * z + 0.5 * v
    e = np.array(errs)
    mask = (e > 1e-11) & (e < e[0] * 0.5)
    ks = np.arange(len(e))[mask]
    slope = float(np.exp(np.polyfit(ks, np.log(e[mask]),
                                    1)[0])) if len(ks) > 10 else float('nan')
    print(f'noiseless outer rate: measured {slope:.4f}  vs  q-tilde bound '
          f'{qt:.4f} (q = {q:.4f}, mu_Gamma = {muG:.3f})')
    # (b) noise linearity
    rows = []
    for psnr in cfg.ou_psnrs:
        sig = np.exp(-psnr / 10.0)
        fin = []
        for _ in range(cfg.ou_trials):
            eps = sig * (rng.standard_normal(L) +
                         1j * rng.standard_normal(L)) / np.sqrt(2)
            yb = G @ xstar + eps
            xh, _ = gcpgd(yb,
                          G,
                          pinv(G) @ yb,
                          N,
                          P,
                          K,
                          w,
                          n_cadzow=cfg.ou_ncad,
                          alpha=0.5,
                          max_iter=cfg.ou_maxit,
                          tol=1e-8)
            fin.append((float(np.linalg.norm(eps)), gn(xh - xstar)))
        rows += [(psnr, ne, er) for ne, er in fin]
    _write_csv(cfg, 'outer_noise', ['psnr', 'eps_norm', 'err_gamma'], rows)
    ne = np.array([r[1] for r in rows])
    er = np.array([r[2] for r in rows])
    good = (ne > 0) & (er > 0)
    ll = np.polyfit(np.log(ne[good]), np.log(er[good]), 1)
    print(f'noise linearity: log-log slope {ll[0]:.3f} (theory: 1.0); '
          f'prefactor {np.exp(ll[1]):.3f} vs bound 2/mu = {2/muG:.3f}')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'plots', 'custom_style.mplstyle'))
        fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.0))
        axs[0].semilogy(e, color='#348ABD', lw=2.2, label='measured')
        axs[0].set_xlabel('outer iteration $k$', fontsize=11)
        axs[0].set_ylabel(r'$\|x_k - x^\ast\|_\Gamma$', fontsize=11)
        axs[0].semilogy(np.arange(len(e)),
                        e[0] * qt**np.arange(len(e)),
                        color='#E24A33',
                        ls='--',
                        lw=1.8,
                        label=r'$\tilde q^{\,k}$ bound')
        axs[0].legend(fontsize=9, loc='upper right')
        axs[1].loglog(ne[good],
                      er[good],
                      'o',
                      color='#348ABD',
                      ms=5,
                      alpha=0.8,
                      label='measured')
        gr = np.array([ne[good].min(), ne[good].max()])
        axs[1].loglog(gr, (2 / muG) * gr,
                      color='#E24A33',
                      ls='--',
                      lw=1.8,
                      label=r'$(2/\mu)\|\epsilon\|$')
        axs[1].set_xlabel(r'$\|\epsilon\|_2$', fontsize=11)
        axs[1].set_ylabel(r'$\|\bar x - x^\ast\|_\Gamma$', fontsize=11)
        axs[1].legend(fontsize=9, loc='upper left')
        fig.tight_layout()
        fig.savefig(_p(cfg, 'outer.pdf'), bbox_inches='tight', pad_inches=0.01)
        print('figure ->', _p(cfg, 'outer.pdf'))
    except Exception as e2:
        print('plot skipped:', e2)


# --------------------------------------------------------------------------- #
#  lipschitz
# --------------------------------------------------------------------------- #
def cmd_lipschitz(cfg, rng=None):
    r"""Lipschitz constant of Cadzow Denoising (Section IV.A)."""
    if rng is None:
        rng = np.random.default_rng(cfg.seeds[0])
    from collections import defaultdict
    lip_const_dict = defaultdict(dict)
    rows = []

    for P in cfg.lip_Ps:
        N = 2 * P + 1
        w = toeplitz_weights(N, P)
        for K in cfg.lip_Ks:
            if K > P:
                continue

            # Sample the hypercube randomly
            low = -1.0
            diameter = 2.0

            random_samps = (rng.uniform(0, 1, (cfg.lip_runs, N)) +
                            1j * rng.uniform(0, 1, (cfg.lip_runs, N)))
            random_samps_2 = (rng.uniform(0, 1, (cfg.lip_runs, N)) +
                              1j * rng.uniform(0, 1, (cfg.lip_runs, N)))

            samples = low + diameter * random_samps / np.abs(random_samps)
            samples_2 = low + diameter * random_samps_2 / np.abs(
                random_samps_2)

            lip_const_vals = []
            for n_run in range(cfg.lip_runs):
                fs_hat = cadzow_denoiser(samples[n_run], N, P, K, cfg.lip_ncad,
                                         w)
                fs_hat_2 = cadzow_denoiser(samples_2[n_run], N, P, K,
                                           cfg.lip_ncad, w)

                num = np.linalg.norm(fs_hat_2 - fs_hat)
                den = np.linalg.norm(samples_2[n_run] - samples[n_run])
                val = num / den if den > 1e-12 else 0.0
                lip_const_vals.append(val)
                rows.append((P, K, n_run, val))

            lip_const_dict[P][K] = lip_const_vals
            print(
                f"P = {P} - K = {K} - Estimated Lip. const: {np.mean(lip_const_vals):.4f}"
            )

    _write_csv(cfg, 'lipschitz', ['P', 'K', 'run', 'lip_val'], rows)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'plots', 'custom_style.mplstyle'))

        plot_Ps = [P for P in cfg.lip_Ps if P > 2]
        n_plots = len(plot_Ps)
        if n_plots > 0:
            fig, axs = plt.subplots(1,
                                    n_plots,
                                    figsize=(4.5 * n_plots, 4.0),
                                    squeeze=False)
            for i, P in enumerate(plot_Ps):
                ax = axs[0, i]
                k_list = sorted(lip_const_dict[P].keys())
                vals = [lip_const_dict[P][k] for k in k_list]
                ax.boxplot(vals, positions=k_list)
                ax.axhline(np.sqrt(P + 1),
                           linestyle='--',
                           color='#E24A33',
                           lw=1.8,
                           label=r'$\sqrt{P+1}$')
                ax.set_title(f"$P={P}$", fontsize=12)
                ax.set_xlabel("$K$", fontsize=11)
                if i == 0:
                    ax.set_ylabel("$H_n$", fontsize=11)
                if i == n_plots - 1:
                    ax.legend(fontsize=9, loc='upper left')
            fig.tight_layout()
            fig.savefig(_p(cfg, 'lipschitz.pdf'),
                        bbox_inches='tight',
                        pad_inches=0.01)
            print('figure ->', _p(cfg, 'lipschitz.pdf'))
    except Exception as e:
        print('plot skipped:', e)


# --------------------------------------------------------------------------- #
#  simulation helpers & algorithm implementations
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#  simulation
# --------------------------------------------------------------------------- #
def cmd_simulation(cfg, rng=None):
    r"""Reconstruction accuracy of CPGD, GCPGD, and GenFRI (Section V.A)."""
    import time
    import warnings
    import scipy.linalg as splin

    rows = []
    K = cfg.sim_K

    rng_dirac = np.random.default_rng(cfg.sim_seed)
    grid = np.arange(0.01, 0.99, 0.01 * (0.99 - 0.01))
    perm = rng_dirac.permutation(len(grid))
    locations = np.sort(grid[perm[:K]])
    intensities = rng_dirac.lognormal(mean=0.0, sigma=0.5, size=K)

    results_data = {
        b: {
            ps: {
                'CPGD': [],
                'GCPGD': [],
                'GenFRI': []
            }
            for ps in cfg.sim_psnrs
        }
        for b in cfg.sim_beta
    }

    for b_idx, beta_val in enumerate(cfg.sim_beta):
        M = beta_val * K
        P = M
        N = 2 * M + 1
        L = 2 * K + 1
        w = toeplitz_weights(N, P)

        rng_samp = np.random.default_rng(1)
        grid_samp = np.arange(0, 1, 0.005)
        sampling_locations = np.sort(grid_samp[rng_samp.permutation(
            len(grid_samp))[:L]])

        frequencies = np.arange(-M, M + 1)
        G = np.exp(2j * np.pi * np.outer(sampling_locations, frequencies))

        fs_coeff = fri_fourier(locations, intensities, M)
        data_noiseless = G @ fs_coeff

        print(f"********** N={N}, L={L} **********")

        for _sd in cfg.seeds:
            rng_trial = np.random.default_rng(_sd)
            for ps in cfg.sim_psnrs:
                noise_lvl = np.max(intensities) * np.exp(-ps / 10.0)

                for trial_idx in range(cfg.sim_trials):
                    std_noise = rng_trial.standard_normal(L)
                    data_noisy = data_noiseless + noise_lvl * std_noise

                    rho = np.linalg.norm(data_noisy) if (b_idx
                                                         == len(cfg.sim_beta) -
                                                         1) else np.inf
                    x_cpgd, iter_cpgd, time_cpgd = run_cpgd(
                        data_noisy,
                        G,
                        N,
                        P,
                        K,
                        w,
                        cfg.sim_ncad,
                        max_iter=cfg.sim_maxit,
                        tol=1e-7,
                        rho=rho)
                    err_cpgd = float(np.linalg.norm(x_cpgd - fs_coeff))
                    loc_cpgd = recover_locations(x_cpgd, N, P, K)
                    pos_err_cpgd = average_match_error(locations, loc_cpgd)

                    results_data[beta_val][ps]['CPGD'].append(err_cpgd)
                    rows.append((beta_val, ps, _sd, trial_idx, 'CPGD',
                                 err_cpgd, pos_err_cpgd, time_cpgd, iter_cpgd))

                    x0_gcpgd = np.zeros(N, dtype=complex)
                    t_start_gcpgd = time.time()
                    x_gcpgd, _, iter_gcpgd = gcpgd(data_noisy,
                                                   G,
                                                   x0_gcpgd,
                                                   N,
                                                   P,
                                                   K,
                                                   w,
                                                   n_cadzow=cfg.sim_ncad,
                                                   alpha=0.5,
                                                   max_iter=cfg.sim_maxit,
                                                   tol=1e-7,
                                                   return_iter=True)
                    time_gcpgd = time.time() - t_start_gcpgd
                    err_gcpgd = float(np.linalg.norm(x_gcpgd - fs_coeff))
                    loc_gcpgd = recover_locations(x_gcpgd, N, P, K)
                    pos_err_gcpgd = average_match_error(locations, loc_gcpgd)

                    results_data[beta_val][ps]['GCPGD'].append(err_gcpgd)
                    rows.append(
                        (beta_val, ps, _sd, trial_idx, 'GCPGD', err_gcpgd,
                         pos_err_gcpgd, time_gcpgd, iter_gcpgd))

                    x_genfri, iter_genfri, time_genfri = run_genfri(
                        data_noisy,
                        G,
                        N,
                        P,
                        K,
                        max_iter=50,
                        nb_init=15,
                        tol=1e-6,
                        rcond=1e-4,
                        seed=_sd)
                    err_genfri = float(np.linalg.norm(x_genfri - fs_coeff))
                    loc_genfri = recover_locations(x_genfri, N, P, K)
                    pos_err_genfri = average_match_error(locations, loc_genfri)

                    results_data[beta_val][ps]['GenFRI'].append(err_genfri)
                    rows.append(
                        (beta_val, ps, _sd, trial_idx, 'GenFRI', err_genfri,
                         pos_err_genfri, time_genfri, iter_genfri))

                m_c = np.median(results_data[beta_val][ps]['CPGD'])
                m_g = np.median(results_data[beta_val][ps]['GCPGD'])
                m_f = np.median(results_data[beta_val][ps]['GenFRI'])
                print(
                    f"PSNR = {ps:3d} dB | Median Coeff Err: CPGD={m_c:.4f}, GCPGD={m_g:.4f}, GenFRI={m_f:.4f}"
                )

    _write_csv(cfg, 'simulation', [
        'beta', 'psnr', 'seed', 'trial', 'method', 'err_coeff', 'err_loc',
        'time', 'iter'
    ], rows)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'plots', 'custom_style.mplstyle'))

        cmap = matplotlib.colormaps.get_cmap("tab10")

        for beta_val in cfg.sim_beta:
            fig, ax = plt.subplots(figsize=(5.5, 4.2))

            methods = ['GenFRI', 'CPGD', 'GCPGD']
            markers = {'GenFRI': 'o', 'CPGD': 'D', 'GCPGD': 's'}
            colors = {'GenFRI': cmap(4), 'CPGD': cmap(0), 'GCPGD': cmap(6)}

            for method in methods:
                medians = []
                p25 = []
                p75 = []
                for ps in cfg.sim_psnrs:
                    vals = np.array(results_data[beta_val][ps][method])
                    medians.append(100.0 * np.median(vals))
                    p25.append(100.0 * np.percentile(vals, 25))
                    p75.append(100.0 * np.percentile(vals, 75))

                ax.plot(cfg.sim_psnrs,
                        medians,
                        '-',
                        marker=markers[method],
                        color=colors[method],
                        label=method,
                        ms=6,
                        lw=2.2)
                ax.fill_between(cfg.sim_psnrs,
                                p25,
                                p75,
                                color=colors[method],
                                alpha=0.30,
                                lw=0)

            ax.set_yscale('log')
            ax.set_xlabel('PSNR (dB)', fontsize=11)
            ax.set_ylabel(r'Reconstruction error (\%)', fontsize=11)
            ax.set_title(r'$\beta = ' + f'{beta_val}' + r'$', fontsize=12)
            ax.legend(fontsize=10)

            fig.tight_layout()
            fig.savefig(_p(cfg, f'reconstruction_error_beta_{beta_val}.pdf'),
                        bbox_inches='tight',
                        pad_inches=0.01)
            print('figure ->',
                  _p(cfg, f'reconstruction_error_beta_{beta_val}.pdf'))

    except Exception as e:
        print('plot skipped:', e)


# --------------------------------------------------------------------------- #
#  infra
# --------------------------------------------------------------------------- #
def _p(cfg, name):
    sfx = getattr(cfg, '_sfx', '')
    if sfx and '.' in name:
        stem, ext = name.rsplit('.', 1)
        name = f'{stem}{sfx}.{ext}'
    elif sfx:
        name = name + sfx
    return os.path.join(cfg.outdir, name)


def _write_csv(cfg, name, header, rows):
    path = _p(cfg, name + '.csv')
    with open(path, 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(header)
        for r in rows:
            wcsv.writerow(r)
    print('csv ->', path)


def parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd',
                   choices=[
                       'rate', 'geometry', 'phase', 'outer', 'vanilla',
                       'certificate', 'all', 'phase-aggregate', 'lipschitz',
                       'simulation'
                   ])
    p.add_argument('--full', action='store_true')
    p.add_argument('--seed',
                   default='0',
                   help='integer seed, or comma-separated list '
                   'for in-process multi-seed pooling')
    p.add_argument('--outdir', default='.')
    p.add_argument(
        '--inputs',
        default='',
        help=
        'comma-separated dirs/globs of per-seed phase runs (phase-aggregate)')
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    a.seeds = [int(x) for x in str(a.seed).split(',')]
    fast = not a.full
    # rate
    a.rate_KM = [(3, 8), (5, 10)] if fast else [(3, 8), (5, 10), (7, 12),
                                                (9, 18)]
    # rate_deltas capped so K*delta < 1 for the largest K (=9 in --full).
    a.rate_deltas = [0.1, 0.05] if fast else [0.10, 0.08, 0.05, 0.035, 0.025]
    a.rate_trials = 3 if fast else 20
    # geometry
    # oversample (M large vs K) so the Lemma 4 validity window 1/Delta <
    # min(N-P,P+1)-1 covers most of the plotted Delta range; the grid spans
    # from near the 1/K feasibility limit down through the validity edge.
    a.geo_K, a.geo_M = (3, 12) if fast else (4, 20)
    # geo_deltas upper end capped so K*delta < 1 (feasibility of sample_locations):
    # fast K=3 -> <1/3; full K=4 -> <1/4.
    a.geo_deltas = list(
        np.round(np.geomspace(0.05, 0.30 if fast else 0.24, 6 if fast else 14),
                 4))
    a.geo_trials = 20 if fast else 100
    # phase
    a.ph_K, a.ph_M = (3, 16) if fast else (5, 35)
    a.ph_gaps = list(np.round(np.geomspace(0.005, 0.24, 5 if fast else 10), 4))
    a.ph_psnrs = (list(range(-10, 41, 5)) if fast else list(range(-10, 56, 3)))
    a.ph_trials = 6 if fast else 40
    a.ph_ncad, a.ph_maxit = 4, (900 if fast else 2500)
    # certificate (Remark 2)
    a.ce_KM = [(2, 6), (3, 8), (4, 9), (5, 12)] if fast else \
        [(2, 6), (3, 8), (3, 12), (4, 9), (5, 12), (5, 16)]
    a.ce_deltas = [0.05, 0.08, 0.12, 0.18] if not fast else [0.07, 0.12]
    a.ce_trials = 5 if fast else 25
    a.ce_ncyc = 12 if fast else 20
    # vanilla (G = Id)
    a.va_K, a.va_M = (3, 12) if fast else (5, 16)
    a.va_delta = 0.08 if fast else 0.05
    a.va_psnrs = np.arange(-10, 55, 5)
    a.va_trials = 12 if fast else 60
    a.va_ncad, a.va_ncad_oneshot = 5, 25
    a.va_maxit = 400 if fast else 4000

    # outer
    a.ou_K, a.ou_M = (3, 8) if fast else (5, 12)
    a.ou_delta = 0.15 if fast else 0.10
    a.ou_iters = 250 if fast else 800
    a.ou_ncad = 5
    a.ou_psnrs = [0, 10, 20, 30, 40
                  ] if fast else [0, 5, 10, 15, 20, 25, 30, 35, 40]
    a.ou_trials = 4 if fast else 20
    a.ou_maxit = 1500 if fast else 4000

    # lipschitz
    a.lip_Ps = np.arange(2, 6, 2) if fast else np.arange(2, 12, 2)
    a.lip_Ks = np.arange(2, 12)
    a.lip_runs = 10 if fast else 100
    a.lip_ncad = 10

    # simulation
    a.sim_K = 5 if fast else 9
    a.sim_beta = [1, 2] if fast else [1, 2, 3]
    a.sim_psnrs = [-10, 0, 10, 20] if fast else [-30, -20, -10, 0, 10, 20, 30]
    a.sim_trials = 4 if fast else 50
    a.sim_ncad = 5
    a.sim_seed = 4
    a.sim_maxit = 4000

    return a


if __name__ == '__main__':
    cfg = parse()
    rng = np.random.default_rng(cfg.seeds[0])
    cmds = {
        'rate': cmd_rate,
        'geometry': cmd_geometry,
        'phase': cmd_phase,
        'outer': cmd_outer,
        'vanilla': cmd_vanilla,
        'certificate': cmd_certificate,
        'phase-aggregate': cmd_phase_aggregate,
        'lipschitz': cmd_lipschitz,
        'simulation': cmd_simulation,
    }
    if cfg.cmd == 'all':
        for k in ['geometry', 'rate', 'outer', 'phase']:
            print(f'===== {k} =====')
            if k == 'outer' and len(cfg.seeds) > 1:
                for _sd in cfg.seeds:
                    cfg._sfx = f'_s{_sd}'
                    cmds[k](cfg, np.random.default_rng(_sd))
                cfg._sfx = ''
            else:
                cmds[k](cfg, rng)
    elif cfg.cmd == 'outer' and len(cfg.seeds) > 1:
        for _sd in cfg.seeds:
            cfg._sfx = f'_s{_sd}'
            cmds['outer'](cfg, np.random.default_rng(_sd))
        cfg._sfx = ''
    else:
        cmds[cfg.cmd](cfg, rng)
    print('done.')
