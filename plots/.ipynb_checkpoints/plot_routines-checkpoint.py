# ############################################################################
# plot_routines.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# ############################################################################
"""
Plotting routines.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom_style.mplstyle'))
cmap = plt.get_cmap("tab10")

def profiles_with_quantiles(x: np.ndarray, median: np.ndarray, percentile_bottom: np.ndarray,
                            percentile_top: np.ndarray,
                            cmap, color_ind: tuple, markers: list, algo_names: list, markersize: int = 6,
                            linewidth: int = 2, alpha: float = 0.3, xlabel: str = 'PSNR (dB)',
                            ylabel: str = 'Positioning Error (\% of $T$)'):
    """
    Performance plots (median + inter-percentile distance).
    :param x: np.ndarray[n,]
    Absciss.
    :param median: np.ndarray[n,k]
    Median of ordinates for each algorithm.
    :param percentile_bottom: np.ndarray[n,k]
    Lower percentile of ordinates for each algorithm.
    :param percentile_top: np.ndarray[n,k]
    Upper percentile of ordinates for each algorithm.
    :param cmap: cmap
    Colormap object.
    :param color_ind: tuple
    Colors from the colormap to be used.
    :param markers: str
    Marker supported by `matplotlib`.
    :param algo_names: list[str] of size k
    Names of the algorithms.
    :param markersize: int
    :param linewidth: float
    :param alpha: float
    :param xlabel: str
    :param ylabel: str
    """

    for k, algo_name in enumerate(algo_names):
        plt.plot(x, median[:, k], '-', fillstyle='full', markersize=markersize, linewidth=linewidth,
                 color=cmap(color_ind[k]), marker=markers[algo_name], label=algo_name)
        plt.fill_between(x, percentile_bottom[:, k], percentile_top[:, k], linewidth=0, color=cmap(color_ind[k]),
                         alpha=alpha)
    plt.legend(fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.xlim((np.min(x), np.max(x)))
    plt.ylabel(ylabel, fontsize=16)


def simu_plots(settings: str, results: str, save_folder: str, percentile: int = 25, cmap_name: str = 'tab10',
               color_ind: tuple = (4, 0, 6)):
    """
    Routine for plotting the various diagnostic plots from [Section V,1].
    :param settings: str
    Path to the pickle file containing the settings of the simulation.
    :param results: str
    Path to the pickle file containing the results of the simulation.
    :param save_folder: str
    Path to the folder where plots should be saved.
    :param percentile: int
    Percentile to be used for inter-percentiles regions.
    :param cmap_name: str
    Name of colormap.
    :param color_ind: tuple
    Which colors from the colormap are used.

    Reference:
    [1] Simeoni, M., Besson, A., Hurley, P. & Vetterli, M. (2020). Cadzow Plug-and-Play Gradient Descent for Generalised FRI.
    Under review.
    """
    # Load simulation settings
    with open(settings, 'rb') as file:
        settings_dict = pickle.load(file)
    # settings_dirac = settings_dict['dirac']
    settings_experiment = settings_dict['experiment']
    algo_names = settings_dict['algo_names']
    algo_markers = settings_dict['algo_markers']
    PSNR = settings_experiment['PSNR']
    beta = settings_experiment['beta']
    K = settings_experiment['K']
    M = settings_experiment['M']
    N = settings_experiment['N']
    L = settings_experiment['L']
    period = settings_experiment['period']
    nb_exp = settings_experiment['nb_exp']

    # Load results
    with open(results, 'rb') as file:
        results_dict = pickle.load(file)
    store_results = results_dict['store_results']

    # Compute median and quatiles of the results
    median_results = np.median(store_results, axis=2)
    percentile_bottom_results = np.percentile(store_results, percentile, axis=2)
    percentile_top_results = np.percentile(store_results, 100 - percentile, axis=2)

    # Decide on colors:
    cmap = plt.get_cmap(cmap_name)

    for b in range(beta.size):
        
        # Reconstruction Error
        plt.figure()
        profiles_with_quantiles(PSNR, 100 * median_results[b], 100 * percentile_bottom_results[b],
                                100 * percentile_top_results[b], cmap,
                                color_ind, algo_markers, algo_names, markersize=6, linewidth=2, alpha=0.3,
                                xlabel='PSNR (dB)', ylabel='Reconstruction error')
        plt.ylim((np.min(100 * percentile_bottom_results[b]),
                  np.max(100 * percentile_top_results[b])))
        plt.yscale('log')
        plt.savefig(
            os.path.join(save_folder, f'reconstruction_error_irr_samp_K={K}_N={N[b]}_L={L[b]}_rep={nb_exp}.pdf'),
            dpi=600, transparent=False, bbox_inches='tight')
