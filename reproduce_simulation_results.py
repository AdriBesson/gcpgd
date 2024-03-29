# ############################################################################
# reproduce_simulation_results.py
# =======
# Authors : Adrien Besson [adribesson@gmail.com] and Matthieu Simeoni [matthieu.simeoni@gmail.com]
# ############################################################################
"""
This script assesses the performances (positioning errors, execution times, number of iterations) of LS-Cadzow, CPGD and
GenFRI for various oversampling parameters, peak signal-to-noise (PSNR) ratios and noise levels.
This is also the script for reproducing the results of [Section V.A, 1].
* Results of the simulations are saved in the folder `../results`, in the subfolder specified by `save_folder`.
* For re-running the simulations, set `run_simu` to True.
* For re-generating the plots of past simulations, set `run_simu` to False and specify in `save_folder` the name of the
subfolder in `../results` containing the results.

Note: Although ran in parallel, simulations may still take a few hours to execute on a laptop! To reduce computation
time, consider setting `nb_exp` to a smaller number (e.g. 12 instead of the default 192).

Reference:
[1] Simeoni, M., Besson, A., Hurley, P. & Vetterli, M. (2020). Cadzow Plug-and-Play Gradient Descent for Generalised FRI.
Under review.
"""
import pickle
import numpy as np
from joblib import Parallel, delayed
import pyoneer.model.dirac_stream as mod
from plots.plot_routines import simu_plots
from pyoneer.operators.linear_operator import ToeplitzificationOperator, FRISampling, LinearOperatorFromMatrix
from scipy.linalg import lstsq
from pyoneer.utils.fri import coeffs_to_matched_diracs
from pyoneer.algorithms.cadzow_denoising import CadzowAlgorithm
from pyoneer.algorithms.genfri import GenFRIAlgorithm
from pyoneer.algorithms.cpgd import CPGDAlgorithm
from pyoneer.algorithms.gcpgd import GCPGDAlgorithm
from scipy.sparse.linalg import eigs
import time
import datetime
import os

import sys
sys.path.append('./')
sys.path.append('pyoneer/.')


sys.path.append('./')
sys.path.append('pyoneer/.')


def algorithmic_contest(data_noisy: np.ndarray, G: np.ndarray, K: int, period: np.float32,
                        locations: np.ndarray, algo_names: list, settings_cadzow: dict,
                        settings_cpgd: dict, settings_genfri: dict, settings_chsd: dict, data_ref=None):
    # Input and output local variables
    results = np.zeros(shape=(len(algo_names),))
    recovery_error = np.zeros(shape=(len(algo_names),))
    positions = np.zeros(shape=(len(algo_names), K))
    times = np.zeros(shape=(len(algo_names),))
    iters = np.zeros(shape=(len(algo_names),))

    for i, algo_name in enumerate(algo_names):
        # CPGD
        if algo_name == 'CPGD':
            cpgd = CPGDAlgorithm(**settings_cpgd)
            fs_coeff_recovered = cpgd.reconstruct(data_noisy, verbose=False)
            total_iterations = cpgd.total_iterations
            total_time = cpgd.total_time
        # Generalized CPGD
        elif algo_name == 'GCPGD':
            gcpgd = GCPGDAlgorithm(**settings_gcpgd)
            fs_coeff_recovered = gcpgd.reconstruct(data_noisy, verbose=False)
            total_iterations = gcpgd.total_iterations
            total_time = gcpgd.total_time
        # GenFRI
        elif algo_name == 'GenFRI':
            genfri = GenFRIAlgorithm(**settings_genfri)
            fs_coeff_recovered = genfri.reconstruct(data_noisy, verbose=False)
            total_iterations = genfri.total_iterations
            total_time = genfri.total_time
        estimated_locations, cost = coeffs_to_matched_diracs(
            fs_coeff_recovered, K, period, locations)

        times[i] = float(total_time)
        results[i] = np.linalg.norm(fs_coeff_recovered - fs_coeff)
        positions[i] = estimated_locations.astype(float)
        iters[i] = int(total_iterations)

    return results, positions, times, iters


if __name__ == '__main__':
    run_simu = True  # bool, Re-run simulations
    # str, name of pickle file storing settings
    filename_general_settings = 'general_settings.pickle'
    filename_results = 'results.pickle'  # str, name of pickle file storing results
    # datetime.datetime.now().strftime("%d%m%Y-%H%M%S")  # str, name of folder in which to save the results.
    save_folder = datetime.datetime.now().strftime(
        "%d%m%Y-%H%M%S")  # '14112023-195344'
    # The results of the paper are saved in the folder  `../results/paper_simulation_results`.

    # Check if results folder exists or create it:
    cwd = os.getcwd()
    results_dir = os.path.join(cwd, 'results', save_folder)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    if run_simu:
        # Parameters
        K = 9  # int, number of Diracs
        beta = np.array([1, 2, 3])  # np.ndarray, oversampling parameter
        M = beta * K  # np.ndarray, bandwidth parameter
        P = M  # np.ndarray, parameter P in [1].
        # np.ndarray, sizes of seeked Fourier series coefficients.
        N = 2 * M + 1
        L = [2 * K + 1]*len(beta)  # float, number of measurements
        period = 1  # float, period of Dirac stream
        # np.ndarray, peak signal-to-noise ratios.
        PSNR = list(np.linspace(-30, 30, 7))
        nb_exp = 50  # int, number of random noise realizations.
        seed = 4  # int, seed of random number generator for reproducibility.
        tol = 1e-7  # float, tolerance for stopping criterion.
        # float, tolerance for low-rank approximation if `backend_cadzow` is 'scipy.sparse'.
        eig_tol = 1e-9
        # int, number of iterations in Cadzow denoising (typically smaller than 20).
        nb_cadzow_iter = 5
        backend_cadzow = 'scipy'  # str,  backend for low-rank approximation.

        algo_markers = {'GenFRI': 'o', 'CPGD': 'D', 'GCPGD': 's'}
        algo_names = list(algo_markers.keys())
        nb_algorithms_in_contest = len(algo_names)
        # Settings dictionaries used as inputs to algorithms and routines
        settings_dirac = {'dirac_count': K, 'sigma_intensity': 0.5, 't_end': (1 - 0.01) * period,
                          't_start': 0.01 * period,
                          'mean_intensity': 0, 'relative_minimal_distance': 0.01,
                          'intensity_distribution': 'lognormal',
                          'seed': seed}
        settings_cpgd = {'nb_iter': 2000, 'rank': K, 'nb_cadzow_iter': nb_cadzow_iter, 'denoise_verbose': False,
                         'nb_init': 1, 'tol': tol, 'eig_tol': eig_tol, 'tau_init_type': 'safest',
                         'random_state': seed, 'cadzow_backend': backend_cadzow}
        settings_gcpgd = {'nb_iter': 2000, 'rank': K, 'nb_cadzow_iter': nb_cadzow_iter, 'denoise_verbose': False,
                          'alpha': 0.5, 'tau_decrease_law': 'diminishing', 'tol': tol, 'eig_tol': eig_tol, 'tau_init_type': 'safest',
                          'random_state': seed, 'cadzow_backend': backend_cadzow}

        settings_genfri = {'nb_iter': 50, 'nb_init': 15,
                           'tol': tol, 'random_state': seed, 'rcond': 1e-4}
        settings_cadzow = {'nb_iter': nb_cadzow_iter,
                           'rank': K, 'tol': eig_tol, 'backend': backend_cadzow}
        settings_experiment = {'K': K, 'beta': beta, 'M': M, 'P': P, 'N': N, 'L': L, 'period': period, 'PSNR': PSNR,
                               'nb_exp': nb_exp, 'seed': seed}
        settings_joblib = {'n_jobs': -1, 'backend': 'multiprocessing',
                           'verbose': 3}  # For silent multiprocessing, set 'verbose' to 0.

        # Save settings
        with open(os.path.join(results_dir, filename_general_settings), 'wb') as file:
            pickle.dump(dict(genfri=settings_genfri, cpgd=settings_cpgd, cadzow=settings_cadzow, gcpgd=settings_gcpgd,
                             dirac=settings_dirac, experiment=settings_experiment, algo_names=algo_names,
                             algo_markers=algo_markers, joblib=settings_joblib), file)

        # Generate signal innovations
        locations, intensities = mod.rnd_innovations(**settings_dirac)

        # Create the output variables
        store_results = np.zeros(
            shape=(len(beta), len(PSNR), nb_exp, nb_algorithms_in_contest))
        store_recovery_error = np.zeros(
            shape=(len(beta), len(PSNR), nb_exp, nb_algorithms_in_contest))
        store_positions = np.zeros(shape=(len(beta), len(
            PSNR), nb_exp, nb_algorithms_in_contest, K))
        store_times = np.zeros(shape=(len(beta), len(
            PSNR), nb_exp, nb_algorithms_in_contest))
        store_iters = np.zeros(shape=(len(beta), len(
            PSNR), nb_exp, nb_algorithms_in_contest))
        cond_numbers_list = []
        fs_coeff_list = []
        data_noiseless_list = []
        low_pass_signal_list = []

        rng = np.random.RandomState(seed=1)

        # Run simulations for various values of beta:
        for n in range(N.size):
            print(f'********** N={N[n]}, L={L[n]} **********')

            # Generate sampling locations and standardised noise.
            sampling_locations, _ = mod.rnd_innovations(L[n], t_end=period,
                                                        relative_minimal_distance=0.005, seed=1)
            std_noise = rng.standard_normal(
                size=(sampling_locations.size, nb_exp))

            # Generate noiseless Fourier coefficients
            frequencies = np.arange(-M[n], M[n] + 1)
            fs_coeff = mod.fourier_series_coefficients(
                M[n], locations, intensities, period=period)
            fs_coeff_list.append(fs_coeff)

            # Generate the irregular sampling operator
            G = FRISampling(frequencies, sampling_locations, period)
            cond_numbers_list.append(np.linalg.cond(G.mat))
            print(
                f'Cond. num. of G of size {G.shape}: {cond_numbers_list[-1]:.2f}')
            settings_cpgd['linear_op'] = G
            settings_genfri['linear_op'] = G
            settings_gcpgd['linear_op'] = G

            data_noiseless = G(fs_coeff)
            data_noiseless_list.append(data_noiseless)
            low_pass_signal = mod.sinc_samples(np.linspace(0, period, 1000), source_locs=locations,
                                               intensities=intensities, M=M[n], period=period)
            low_pass_signal_list.append(low_pass_signal)

            # Create Toeplitzification Operator
            Tp = ToeplitzificationOperator(P=P[n], M=M[n])
            settings_cadzow['toeplitz_op'] = Tp
            settings_cpgd['toeplitz_op'] = Tp
            settings_genfri['toeplitz_op'] = Tp
            settings_gcpgd['toeplitz_op'] = Tp

            print(settings_dirac)
            # Run benchmark in parallel with joblib
            with Parallel(**settings_joblib) as parallel:
                for i, psnr in enumerate(PSNR):
                    print(f'Iteration: {i}, PSNR: {psnr} dB')
                    noise_lvl = np.max(intensities) * np.exp(-psnr / 10)
                    data_noisy = data_noiseless[:,
                                                None] + noise_lvl * std_noise
                    if n == N.size - 1:
                        settings_cpgd['rho'] = np.linalg.norm(data_noisy)
                    list_multi = parallel(
                        delayed(algorithmic_contest)(data_noisy[:, k], G.mat, K, period, locations, algo_names,
                                                     settings_cadzow, settings_cpgd, settings_genfri, settings_gcpgd, fs_coeff)
                        for k in range(nb_exp))
                    sublist_results = [list_element[0]
                                       for list_element in list_multi]
                    sublist_positions = [list_element[1]
                                         for list_element in list_multi]
                    sublist_times = [list_element[2]
                                     for list_element in list_multi]
                    sublist_iters = [list_element[-2]
                                     for list_element in list_multi]
                    store_results[n, i] = np.stack(sublist_results, axis=0)
                    store_positions[n, i] = np.stack(sublist_positions, axis=0)
                    store_times[n, i] = np.stack(sublist_times, axis=0)
                    store_iters[n, i] = np.stack(sublist_iters, axis=0)
        # Store the results to be able to regenerate the plots
        with open(os.path.join(results_dir, filename_results), 'wb') as file:
            results = {'store_results': store_results, 'store_positions': store_positions,
                       'store_times': store_times, 'store_iters': store_iters,
                       'data_noiseless_list': data_noiseless_list, 'low_pass_signal_list': low_pass_signal_list,
                       'sampling_locations': sampling_locations, 'dirac_locations': locations,
                       'dirac_intensities': intensities,
                       'fs_coeff_list': fs_coeff_list, 'cond_numbers_list': cond_numbers_list}
            pickle.dump(results, file)

    # Generate and save the plots
    simu_plots(os.path.join(results_dir, filename_general_settings), os.path.join(results_dir, filename_results),
               results_dir, percentile=25, cmap_name='tab10', color_ind=(4, 0, 6, 1))
