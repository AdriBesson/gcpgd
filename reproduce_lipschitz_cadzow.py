# Script used to reproduce the Licpshcitz contant experiment of the Cadzow denoising algorithm in Section IV of the paper
# "New Perspectives on Generalized Finite Rate of Innovation" by A. Besson and M. Simeoni
# Author: Adrien Besson, adribesson@gmail.com
# Date: February 2024

import sys
import os
sys.path.append('./')
sys.path.append('pyoneer/.')
import numpy as np
import matplotlib.pyplot as plt
from pyoneer.operators.linear_operator import ToeplitzificationOperator
from pyoneer.algorithms.cadzow_denoising import CadzowAlgorithm
from collections import defaultdict

# Parameters of the experiments
seed = 4  # int, seed of random number generator for reproducibility.
eig_tol = 1e-8  # float, tolerance for low-rank approximation if `backend_cadzow` is 'scipy.sparse'.
nb_cadzow_iter = 10  # int, number of iterations in Cadzow denoising (typically smaller than 20).
backend_cadzow = 'scipy'  # str,  backend for low-rank approximation.
n_runs = 100
Ps = np.arange(2, 12, 2)
Ks = np.arange(2, 12)

# Computation of estimates of the Lipschitz contant
lip_const_dict = defaultdict()
for P in Ps:
    lip_const_dict[P] = {}
    for K in Ks:
        if K > P:
            continue

        # Parameters of the Cadzow denoising algorithm
        N = 2 * P + 1         
        settings_cadzow = {'nb_iter': nb_cadzow_iter, 'rank': K, 'tol': eig_tol, 'backend': backend_cadzow}

        # Create Toeplitzification Operator
        Tp = ToeplitzificationOperator(P=P, M=P)
        settings_cadzow['toeplitz_op'] = Tp

        # Create Cadzow denoiser
        cadzow = CadzowAlgorithm(**settings_cadzow)

        # Sample the hypercube randomly
        radius = 1
        center = np.zeros((N, )).astype(np.complex128)
        low = center-radius
        diameter = 2 * radius
        random_samps = (np.random.sample((n_runs, N)) + 1j * np.random.sample((n_runs, N)))
        random_samps_2 = (np.random.sample((n_runs, N)) + 1j * np.random.sample((n_runs, N)))
        samples = low + diameter * random_samps / np.abs(random_samps)
        samples_2 = low + diameter * random_samps_2 / np.abs(random_samps_2)
        lip_const_vals = []
        for n_run in range(n_runs):
            # Generate first set of samples
            fs_hat = cadzow.denoise(samples[n_run])
            fs_hat_2 = cadzow.denoise(samples_2[n_run])
            lip_const_vals.append(np.sqrt(np.sum(np.abs(fs_hat_2 - fs_hat)**2)) / np.sqrt(np.sum(np.abs(samples_2[n_run] - samples[n_run])**2)))

        # Collect the estimates and print
        lip_const_dict[P][K] = lip_const_vals
        print("*************** K = {} - P = {} - Estimated Lip. const {} ********************".format(K, P, np.mean(np.array(lip_const_vals))))

# Draw the results
fig, axs= plt.subplots(1, 4, figsize=(20, 5))
for i, ax in enumerate(axs):
    ax.boxplot([n for n in lip_const_dict[Ps[1+i]].values()], positions=np.arange(2, Ps[1+i] + 1))
    ax.axhline(np.sqrt(Ps[1+i]+1), linestyle='--')

    ax.set_title("$P={}$".format(Ps[1+i]))
    ax.set_xlabel("$K$")
    axs[0].set_ylabel("$H_n$")
    ax.grid(visible=False)
plt.show()

fig.savefig("lip_const_test.pdf", format="pdf", bbox_inches="tight")
