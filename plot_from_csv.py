#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_from_csv.py
================
Regenerates all paper figures directly from the CSV files stored in a directory.
Falls back gracefully if some CSV files are missing.
"""

import os
import sys
import csv
import argparse
import numpy as np

# Ensure gcpgd_lib is importable from current directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    print("Error: matplotlib is required to run this script.")
    sys.exit(1)

# Try importing from gcpgd_lib for quick dynamic overlays
try:
    from gcpgd_lib import (
        toeplitz_weights, gnorm_factory, sample_locations, fri_fourier,
        build_toeplitz, measurement_operator, mu_restricted, check_nontangentiality,
        sigmaK_lemma4, cadzow_denoiser, robust_svd as svd
    )
    has_lib = True
except ImportError:
    print("Warning: gcpgd_lib not found. Some dynamic plot elements (collision sweep, noiseless outer rate) will be skipped.")
    has_lib = False


def load_csv(path):
    """Loads a CSV into a list of dictionaries."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def set_style():
    """Sets matplotlib style if the custom stylesheet is available."""
    style_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'custom_style.mplstyle')
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        plt.style.use('ggplot')


# --------------------------------------------------------------------------- #
#  Plotting Functions
# --------------------------------------------------------------------------- #

def plot_rate(indir, outdir):
    csv_path = os.path.join(indir, 'rate.csv')
    rows = load_csv(csv_path)
    if not rows:
        print(f"Skipping rate: {csv_path} not found.")
        return

    pred = np.array([float(r['c2_pred']) for r in rows if r['c2_pred'] != 'nan'])
    meas = np.array([float(r['rate_meas']) for r in rows if r['rate_meas'] != 'nan'])

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot([0, 1], [0, 1], '--', color='#555555', lw=1.2, label='measured = predicted')
    ax.scatter(pred, meas, color='#348ABD', s=20, alpha=0.8)
    ax.set_xlabel(r'predicted rate $c^2$ (certificate)')
    ax.set_ylabel(r'measured per-cycle rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc='upper left')
    fig.tight_layout()
    
    out_path = os.path.join(outdir, 'rate.pdf')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
    print('Regenerated rate plot ->', out_path)


def plot_geometry(indir, outdir, full=False):
    csv_path = os.path.join(indir, 'geometry.csv')
    rows = load_csv(csv_path)
    if not rows:
        print(f"Skipping geometry: {csv_path} not found.")
        return

    ds = np.array(sorted(set(float(r['delta']) for r in rows)))
    
    # Helper lambda to filter by delta
    col_by_delta = lambda col_name, d: [float(r[col_name]) for r in rows if abs(float(r['delta']) - d) < 1e-7]
    med = lambda col_name, d: np.median(col_by_delta(col_name, d))
    q1 = lambda col_name, d: np.percentile(col_by_delta(col_name, d), 25)
    q3 = lambda col_name, d: np.percentile(col_by_delta(col_name, d), 75)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # -- panel A: measured sigma_K (with IQR band) vs Lemma 4 bound --
    m_sig = np.array([med('sigma_K', d) for d in ds])
    lo = np.array([q1('sigma_K', d) for d in ds])
    hi = np.array([q3('sigma_K', d) for d in ds])
    
    axL.fill_between(ds, lo, hi, color='#348ABD', alpha=0.30, label='measured IQR')
    axL.plot(ds, m_sig, 'o-', color='#348ABD', ms=4, label=r'measured median')
    
    b_sig = np.array([
        np.nan if np.all(np.isnan(col_by_delta('sigmaK_bound', d))) else np.nanmedian(col_by_delta('sigmaK_bound', d))
        for d in ds
    ])
    axL.plot(ds, b_sig, 'k--', lw=1.5, label=r'Lemma 4 bound')
    
    # Estimate validity edge from P
    # n_unit = 2K. Let's find max n_unit to get K
    n_units = [int(r['n_unit']) for r in rows if r['n_unit'] != 'nan']
    K = max(n_units) // 2 if n_units else (4 if full else 3)
    M = 20 if K == 4 else 12
    N, P = 2 * M + 1, M
    d_edge = 1.0 / (min(N - P, P + 1) - 1)
    
    axL.axvline(d_edge, color='#E24A33', ls=':', lw=1.5, label=r'validity edge')
    axL.set_xlabel(r'separation $\Delta$', fontsize=20)
    axL.set_xscale('log')
    axL.set_ylabel(r'$\sigma_K$', fontsize=20)
    axL.set_ylim(bottom=0)
    axL.legend(fontsize=15, loc='upper left')
    axL.tick_params(axis='both', which='major', labelsize=16)
    axL.set_title(r'conditioning vs.\ Lemma 4', fontsize=20)

    # -- panel B: collision sweep --
    if has_lib:
        gaps = np.geomspace(0.10, 5e-4, 12)
        c_coll, sk_coll = [], []
        anchors = 0.55
        rng_b = np.random.default_rng(12345)
        geo_trials = 100 if full else 20
        for g in gaps:
            cc, ss = [], []
            for _ in range(max(5, geo_trials // 4)):
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
        axB.set_xlabel(r'colliding-pair gap', fontsize=20)
        axB.invert_xaxis()
        axB.set_ylabel(r'median angle constant $c$', color='#E24A33', fontsize=20)
        axB.set_ylim(0, 1)
        axB.tick_params(axis='x', labelsize=16)
        axB.tick_params(axis='y', labelsize=16, labelcolor='#E24A33')
        axB.axhline(1.0, color='#E24A33', ls=':', lw=1.0, alpha=0.8)
        
        axB2 = axB.twinx()
        axB2.semilogy(gaps, sk_coll, 's-', color='#348ABD', ms=4)
        axB2.set_ylabel(r'median $\sigma_K$ (log)', color='#348ABD', fontsize=20)
        axB2.tick_params(axis='y', labelsize=16, labelcolor='#348ABD')
        axB.set_title(r'collision: $\sigma_K \to 0$ but $c \not\to 1$', fontsize=20)
    else:
        axR.text(0.5, 0.5, 'Panel B requires gcpgd_lib', ha='center', va='center')

    fig.tight_layout()
    out_path = os.path.join(outdir, 'geometry.pdf')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
    print('Regenerated geometry plot ->', out_path)


def plot_phase(indir, outdir):
    csv_path = os.path.join(indir, 'phase.csv')
    meta_path = os.path.join(indir, 'phase_meta.csv')
    rows = load_csv(csv_path)
    meta = load_csv(meta_path)
    if not rows or not meta:
        print(f"Skipping phase: {csv_path} or {meta_path} not found.")
        return

    gaps = np.array(sorted(set(float(r['gap']) for r in rows)))
    psnrs = np.array(sorted(set(float(r['psnr']) for r in rows)))
    
    rate = np.zeros((len(psnrs), len(gaps)))
    for r in rows:
        pi = np.searchsorted(psnrs, float(r['psnr']))
        gi = np.searchsorted(gaps, float(r['gap']))
        rate[pi, gi] = float(r['success'])

    mu_med = np.array([float(m['mu_med']) for m in meta])
    sK_med = np.array([float(m['sK_med']) for m in meta])
    K = int(meta[0]['K'])
    M = int(meta[0]['M'])
    L = int(meta[0]['L'])
    N, P = 2 * M + 1, M
    m1, m2 = N - P, P + 1

    delta_tube = 0.05
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
    
    if has_lib:
        moitra = np.array([sigmaK_lemma4(1.0, N, P, g) for g in gaps])
        th_cor2 = 10 * np.log(np.sqrt(L) / (c2 * mu_med * moitra))
    else:
        th_cor2 = np.full(len(gaps), np.nan)
        
    d_edge = 1.0 / (min(m1, m2) - 1)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    xs = np.arange(len(gaps)) + 0.5
    im = ax.imshow(rate, origin='lower', aspect='auto',
                   extent=[0, len(gaps), psnrs[0], psnrs[-1]],
                   cmap='viridis', vmin=0, vmax=1)
                   
    ax.plot(xs, th_meas, color='#E24A33', ls='-', lw=2.2, label=r'Thm 2 threshold (measured $\sigma_K,\mu$)')
    ax.plot(xs, th_fit, 'w:', lw=1.8, label=f'shape, 1-const fit ({b:+.0f} dB)')
    
    if has_lib:
        ax.plot(xs, th_cor2, color='#348ABD', ls='--', lw=2.0, marker='s', ms=4.5, label='Cor. 2 (explicit, valid)')
        
    if gaps.min() < d_edge < gaps.max():
        xe = float(np.interp(d_edge, gaps, xs))
        ax.axvline(xe, color='#555555', ls=':', lw=1.2, alpha=0.8)
        ax.text(xe, psnrs[0] + 2, ' validity edge', fontsize=8, rotation=90)
        
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{g:.3f}' for g in gaps], fontsize=8)
    ax.set_xlabel(r'minimum separation $\Delta$ (one pair merging)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_ylim(psnrs[0], psnrs[-1])
    ax.set_title(r'exact recovery (err $\leq 0.1\sigma_K$) vs. predicted thresholds', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    fig.colorbar(im, label='success rate')
    
    fig.tight_layout()
    out_path = os.path.join(outdir, 'phase.pdf')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
    print('Regenerated phase plot ->', out_path)


def plot_vanilla(indir, outdir):
    csv_path = os.path.join(indir, 'vanilla.csv')
    rows = load_csv(csv_path)
    if not rows:
        print(f"Skipping vanilla: {csv_path} not found.")
        return

    psnrs = sorted(list(set(float(r['psnr']) for r in rows)))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    metrics = [
        ('err_l2', r'$\ell_2$ error'),
        ('err_gamma', r'$\Gamma$ error'),
        ('err_loc', 'location error')
    ]

    for ai, (col_name, ttl) in enumerate(metrics):
        ax = axes[ai]
        for tag, sty in (('cadzow', 'o-'), ('gcpgd', 's--')):
            med = []
            for ps in psnrs:
                matched_vals = [float(r[col_name]) for r in rows if abs(float(r['psnr']) - ps) < 1e-7 and r['method'] == tag]
                med.append(np.median(matched_vals) if matched_vals else np.nan)
            ax.plot(psnrs, med, sty, ms=5, label=tag)
        
        ax.set_yscale('log')
        ax.set_xlabel('PSNR (dB)', fontsize=11)
        ax.set_title(ttl, fontsize=12)
        
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(outdir, 'vanilla.pdf')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
    print('Regenerated vanilla plot ->', out_path)


def plot_certificate(indir, outdir):
    csv_path = os.path.join(indir, 'certificate.csv')
    rows = load_csv(csv_path)
    if not rows:
        print(f"Skipping certificate: {csv_path} not found.")
        return

    cs = [float(r['c']) for r in rows]
    eh = np.array([float(r['eps_hat']) for r in rows])
    dr = np.array([float(r['drift']) for r in rows])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    a1.scatter(cs, eh, color='#348ABD', s=20, alpha=0.8)
    a1.set_xlabel(r'angle constant $c$', fontsize=11)
    a1.set_ylabel(r'certified $\hat{\varepsilon}$', fontsize=11)
    
    a2.scatter(dr, 1 + eh, color='#348ABD', s=20, alpha=0.8)
    lim = [0, max(1.05, (1 + eh).max() * 1.05)]
    a2.plot(lim, lim, '--', color='#E24A33', lw=1.5, label=r'certified = drift')
    a2.set_xlabel('measured drift', fontsize=11)
    a2.set_ylabel(r'certified bound $1+\hat{\varepsilon}$', fontsize=11)
    a2.legend(fontsize=9, loc='upper left')
    
    fig.tight_layout()
    out_path = os.path.join(outdir, 'certificate.pdf')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
    print('Regenerated certificate plot ->', out_path)


def plot_outer(indir, outdir, seed=0, full=False):
    csv_path = os.path.join(indir, 'outer_noise.csv')
    rows = load_csv(csv_path)
    if not rows:
        print(f"Skipping outer: {csv_path} not found.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # -- panel A: noiseless linear rate (Requires gcpgd_lib) --
    if has_lib:
        cfg_ou_K = 3 if not full else 5
        cfg_ou_M = 8 if not full else 12
        cfg_ou_delta = 0.15 if not full else 0.10
        cfg_ou_iters = 250 if not full else 800
        cfg_ou_ncad = 5

        K, M = cfg_ou_K, cfg_ou_M
        N, P = 2 * M + 1, M
        L = 2 * N
        w = toeplitz_weights(N, P)
        gn = gnorm_factory(w)
        
        rng = np.random.default_rng(seed)
        t = sample_locations(K, cfg_ou_delta, rng)
        a = np.exp(1j * rng.uniform(0, 2 * np.pi, K))
        xstar = fri_fourier(t, a, M)
        sK = float(svd(build_toeplitz(xstar, N, P), compute_uv=False)[K - 1])
        G = measurement_operator(N, L, M, rng)
        muG = mu_restricted(G, w, t, a, M)
        Gs = G * (1.0 / np.sqrt(w))[None, :]
        tau = 1.0 / (2.0 * np.linalg.norm(Gs, 2)**2)
        q = np.sqrt(max(0.0, 1 - 2 * tau * muG**2))
        qt = (1 + q) / 2
        
        y = G @ xstar
        d = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = xstar + 0.2 * sK * d / gn(d)
        errs = []
        Gh = G.conj().T
        for _ in range(cfg_ou_iters):
            errs.append(gn(x - xstar))
            v = x - 2 * tau * ((Gh @ (G @ x - y)) / w)
            z = cadzow_denoiser(v, N, P, K, cfg_ou_ncad, w)
            x = 0.5 * z + 0.5 * v
        e = np.array(errs)

        axs[0].semilogy(e, color='#348ABD', lw=2.2, label='measured')
        axs[0].set_xlabel('outer iteration $k$', fontsize=20)
        axs[0].set_ylabel(r'$\|x_k - x^\ast\|_\Gamma$', fontsize=20)
        axs[0].semilogy(np.arange(len(e)), e[0] * qt**np.arange(len(e)), color='#E24A33', ls='--', lw=1.8, label=r'$\tilde q^{\,k}$ bound')
        axs[0].legend(fontsize=15, loc='upper right')
        axs[0].tick_params(axis='both', which='major', labelsize=16)
    else:
        axs[0].text(0.5, 0.5, 'Panel A requires gcpgd_lib', ha='center', va='center')

    # -- panel B: noise linearity --
    ne = np.array([float(r['eps_norm']) for r in rows])
    er = np.array([float(r['err_gamma']) for r in rows])
    good = (ne > 0) & (er > 0)
    
    axs[1].loglog(ne[good], er[good], 'o', color='#348ABD', ms=5, alpha=0.8, label='measured')
    if len(ne[good]) > 0:
        gr = np.array([ne[good].min(), ne[good].max()])
        # We need muG to draw the threshold. If has_lib wasn't loaded, let's use a dummy muG or skip
        if has_lib:
            axs[1].loglog(gr, (2 / muG) * gr, color='#E24A33', ls='--', lw=1.8, label=r'$(2/\mu)\|\epsilon\|$')
        else:
            axs[1].loglog(gr, 2.0 * gr, color='#E24A33', ls='--', lw=1.8, label=r'$(2/\mu)\|\epsilon\|$ (approx)')
            
    axs[1].set_xlabel(r'$\|\epsilon\|_2$', fontsize=20)
    axs[1].set_ylabel(r'$\|\bar x - x^\ast\|_\Gamma$', fontsize=20)
    axs[1].legend(fontsize=15, loc='upper left')
    axs[1].tick_params(axis='both', which='major', labelsize=16)

    fig.tight_layout()
    out_path = os.path.join(outdir, 'outer.pdf')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
    print('Regenerated outer plot ->', out_path)


def plot_lipschitz(indir, outdir, full=False):
    csv_path = os.path.join(indir, 'lipschitz.csv')
    rows = load_csv(csv_path)
    if not rows:
        print(f"Skipping lipschitz: {csv_path} not found.")
        return

    from collections import defaultdict
    lip_const_dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        P = int(r['P'])
        K = int(r['K'])
        val = float(r['lip_val'])
        lip_const_dict[P][K].append(val)

    lip_Ps = np.arange(2, 6, 2) if not full else np.arange(2, 12, 2)
    plot_Ps = [P for P in lip_Ps if P > 2 and P in lip_const_dict]
    n_plots = len(plot_Ps)
    
    if n_plots > 0:
        fig, axs = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.0), squeeze=False)
        for i, P in enumerate(plot_Ps):
            ax = axs[0, i]
            k_list = sorted(lip_const_dict[P].keys())
            vals = [lip_const_dict[P][k] for k in k_list]
            ax.boxplot(vals, positions=k_list)
            ax.axhline(np.sqrt(P + 1), linestyle='--', color='#E24A33', lw=1.8, label=r'$\sqrt{P+1}$')
            ax.set_title(f"$P={P}$", fontsize=12)
            ax.set_xlabel("$K$", fontsize=11)
            if i == 0:
                ax.set_ylabel("$H_n$", fontsize=11)
            if i == n_plots - 1:
                ax.legend(fontsize=9, loc='upper left')
        
        fig.tight_layout()
        out_path = os.path.join(outdir, 'lipschitz.pdf')
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
        print('Regenerated lipschitz plot ->', out_path)


def plot_simulation(indir, outdir):
    csv_path = os.path.join(indir, 'simulation.csv')
    rows = load_csv(csv_path)
    if not rows:
        print(f"Skipping simulation: {csv_path} not found.")
        return

    # Group results by beta, psnr, and method
    betas = sorted(list(set(int(r['beta']) for r in rows)))
    psnrs = sorted(list(set(int(r['psnr']) for r in rows)))
    methods = ['GenFRI', 'CPGD', 'GCPGD']

    cmap = plt.get_cmap("tab10")
    markers = {'GenFRI': 'o', 'CPGD': 'D', 'GCPGD': 's'}
    colors = {'GenFRI': cmap(4), 'CPGD': cmap(0), 'GCPGD': cmap(6)}

    for beta_val in betas:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))

        for method in methods:
            medians = []
            p25 = []
            p75 = []
            
            for ps in psnrs:
                vals = [float(r['err_coeff']) for r in rows if int(r['beta']) == beta_val and int(r['psnr']) == ps and r['method'] == method]
                if vals:
                    vals = np.array(vals)
                    medians.append(np.median(vals))
                    p25.append(np.percentile(vals, 25))
                    p75.append(np.percentile(vals, 75))
                else:
                    medians.append(np.nan)
                    p25.append(np.nan)
                    p75.append(np.nan)

            ax.plot(psnrs, medians, '-', marker=markers[method], color=colors[method], label=method, ms=6, lw=2.2)
            ax.fill_between(psnrs, p25, p75, color=colors[method], alpha=0.30, lw=0)

        fig.patch.set_facecolor('white')
        ax.set_facecolor('#E5E5E5')
        ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.9)
        ax.set_axisbelow(True)

        ax.set_yscale('log')
        ax.set_xlabel('PSNR (dB)', fontsize=16)
        ax.set_ylabel('Reconstruction error', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.legend(fontsize=15, frameon=True, facecolor='white', edgecolor='#555555')

        fig.tight_layout()
        out_path = os.path.join(outdir, f'reconstruction_error_beta_{beta_val}.pdf')
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.01, transparent=False)
        print(f'Regenerated simulation plot (beta={beta_val}) ->', out_path)


# --------------------------------------------------------------------------- #
#  Main Driver
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Regenerate papers figures from CSV outputs.")
    parser.add_argument('cmd', choices=[
        'rate', 'geometry', 'phase', 'outer', 'vanilla', 'certificate', 'lipschitz', 'simulation', 'all'
    ], help="Which plot to generate.")
    parser.add_argument('--indir', default='results', help="Directory containing the CSV files.")
    parser.add_argument('--outdir', default='results', help="Directory where regenerated figures will be saved.")
    parser.add_argument('--full', action='store_true', help="Uses original '--full' parameter configurations.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for dynamic noiseless outer rate validation.")
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    set_style()
    
    cmds = {
        'rate': lambda: plot_rate(args.indir, args.outdir),
        'geometry': lambda: plot_geometry(args.indir, args.outdir, args.full),
        'phase': lambda: plot_phase(args.indir, args.outdir),
        'vanilla': lambda: plot_vanilla(args.indir, args.outdir),
        'certificate': lambda: plot_certificate(args.indir, args.outdir),
        'outer': lambda: plot_outer(args.indir, args.outdir, args.seed, args.full),
        'lipschitz': lambda: plot_lipschitz(args.indir, args.outdir, args.full),
        'simulation': lambda: plot_simulation(args.indir, args.outdir),
    }

    if args.cmd == 'all':
        for name, fn in cmds.items():
            print(f"\n--- Plotting {name} ---")
            fn()
    else:
        cmds[args.cmd]()

    print("\nPlot regeneration completed.")


if __name__ == '__main__':
    main()
