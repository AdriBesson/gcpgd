# Cadzow Projected Gradient Descent for Generalized Finite Rate of Innovation: A Quantitative Local Convergence Theory
[E-Scopics]: https://www.e-scopics.com/

Adrien Besson<sup>1</sup>, August 2026 

<sup>1</sup>[E-Scopics], France

Code used to reproduce the results presented in the paper entitled *Cadzow Projected Gradient Descent for Generalized Finite Rate of Innovation: A Quantitative Local Convergence Theory*, submitted to IEEE Transactions on Signal Processing

## Abstract
The generalized finite rate of innovation~(GenFRI) framework aims at reconstructing finite-rate-of-innovation~(FRI) signals measured through a noisy linear measurement model.
GenFRI has been recently recast as a structured low-rank optimization problem and the Cadzow projected gradient descent~(CPGD) algorithm has been suggested to solve it.
While CPGD works well in practice, only qualitative local convergence guarantees have been established.
We revisit GenFRI in the light of the regularization by denoising framework, recasting it as an optimization problem whose solutions lie in the fixed-point set of the Cadzow denoiser.
We show that no algorithm in this family can enjoy global guarantees, and establish instead that the Cadzow denoiser is quasi-nonexpansive on an explicit neighborhood of the FRI model set, whose radius is governed by the conditioning of the underlying Dirac stream.
Building on these results, we propose the generalized CPGD~(GCPGD) algorithm and prove its convergence from any initialization within an explicit basin of attraction, together with a reconstruction error bound proportional to the noise level.
We show through numerical simulations that GCPGD outperforms state-of-the-art GenFRI algorithms.

## Repository Structure
The repository has been structured as follows:
* `gcpgd_lib/`: Consolidates all GCPGD algorithms, operators, signal definitions, and certificate checks.
  * `toeplitz.py`: Toeplitz weights, Toeplitz projection, rank projection, and Cadzow denoiser.
  * `signal.py`: FRI location sampling, Fourier coefficients, and measurement operators.
  * `algorithm.py`: GCPGD iteration (using Gamma-gradient step), CPGD, and GenFRI algorithms.
  * `certificates.py`: Nontangentiality certificate, restricted/global mu calculation, and Lemma 4 bounds.
  * `recovery.py`: Annihilating-filter location recovery.
  * `metrics.py`: Circular/average matching errors and Wilson binomial interval utilities.
* `plots/`: Contains custom plotting styles and utilities.
* `reproduce_all_experiments.py`: A unified driver script to run all paper experiments under a clean CLI interface.
* `plot_from_csv.py`: A custom utility script to regenerate all paper figures directly from existing CSV outputs.

## Requirements
* Python environment (Tested on 3.12)
* Python packages: `numpy`, `scipy`, `matplotlib`

## Getting the code

You can download a copy of all the files in this repository by cloning the [git](https://git-scm.com/) repository:

```bash
git clone https://github.com/AdriBesson/gcpgd.git
```

or [download a zip archive](https://github.com/AdriBesson/gcpgd/archive/refs/heads/main.zip).

## Usage
The `reproduce_all_experiments.py` script is the entry point for 1) reproducing all experiments described in the paper; 2) performing some more experiments for exploratory purpose (e.g. looking at the regularity of the Cadzow denoiser through its Lipschitz constant). It supports several subcommands depending on which claim or figure you wish to validate.

### Subcommands
* **`rate`**: Certificate -> rate: measures the per-cycle rate of the inner alternating projections vs. the prediction $c^2$ from the nontangentiality certificate (Definition 2, Lemma 2, Thm 1(iii)).
* **`geometry`**: Ensemble statistics of the certificate: $c$ and $\sigma_K$ vs. the separation $\Delta$; $n_{\text{unit}} = 2K$ throughout (Definition 2 generic).
* **`phase`**: Success-rate phase diagram over $(\text{PSNR}, \Delta)$ on the collision ensemble (one pair merging, $\sigma_K$ collapsing), warm-started at $L = 2N$; overlays the Theorem 2 threshold from measured $(\sigma_K, \mu)$ [mechanism, parallel], its 1-constant calibration [shape hugs], and the explicit Corollary 2 curve on its validity region.
* **`outer`**: Outer-loop validation: noiseless linear rate vs. $\tilde{q}$, and noise-linearity of the limiting error (Theorem 2).
* **`lipschitz`**: Estimates the Lipschitz constant of the Cadzow denoising operator (reproduces Section IV.A results, previously `reproduce_lipschitz_cadzow.py`).
* **`simulation`**: Reconstruction accuracy of GCPGD compared with other algorithms under different noise levels and matrices (reproduces Section IV.B results, previously `reproduce_simulation_results.py`).
* **`vanilla`**: Baseline experiment with $G = I$ (Identity matrix).
* **`certificate`**: Validates certificate properties (Remark 2).
* **`phase-aggregate`**: Aggregates multi-seed phase runs.
* **`all`**: Runs the four main experiments (`geometry`, `rate`, `outer`, `phase`) sequentially.

### Common Options
You can configure each subcommand with the following CLI options:
* `--fast` (default): Runs a quick version of the experiment with fewer trials/grids to verify correctness.
* `--full`: Runs the complete experiment using the dense grids and high trial counts as presented in the paper.
* `--seed <seed_value>`: Sets the RNG seed (or a comma-separated list of seeds for multi-seed pooling).
* `--outdir <dir_path>`: Specifies where output figures (.pdf) and CSVs (.csv) should be saved (defaults to current directory).

### Reproducing Paper Figures
To reproduce the exact same figures as presented in the paper, run the following command:
```bash
python3 reproduce_all_experiments.py all --full --seed 0 --outdir results/
```

> ⚠️ **Warning**: The `phase` experiment (which is included in the `all` sweep) is computationally intensive and can take a significant amount of time to complete.
>
> We strongly recommend running a fast version of the experiment as a sanity check first to verify your setup:
> ```bash
> python3 reproduce_all_experiments.py geometry --seed 0 --outdir results/
> ```

### Additional Examples
To reproduce only the reconstruction accuracy simulation in Section IV.B with full fidelity and save the output plots to a custom folder:
```bash
python3 reproduce_all_experiments.py simulation --full --outdir ./plots_output
```

To run a fast validation of the inner alternating projection rates:
```bash
python3 reproduce_all_experiments.py rate --fast
```

### Regenerating Plots from CSV
If you have already executed the experiments and have the generated CSV outputs in a directory (e.g. `./results`), you can use the lightweight utility `plot_from_csv.py` to regenerate the corresponding paper PDF figures directly without re-running any heavy simulations:

* **Regenerate all figures:**
  ```bash
  python3 plot_from_csv.py all --indir results --outdir results
  ```
* **Regenerate a specific figure (e.g., `geometry`):**
  ```bash
  python3 plot_from_csv.py geometry --indir results --outdir results
  ```
* **Print all available options:**
  ```bash
  python3 plot_from_csv.py --help
  ```

## Contact
Adrien Besson (adribesson@gmail.com)

## License
Please cite the following paper when using the code:
A. Besson, "Cadzow Projected Gradient Descent for Generalized Finite Rate of Innovation: A Quantitative Local Convergence Theory", submitted to IEEE Transactions on Signal Processing 

```
MIT License

Copyright (c) 2026 Adrien BESSON

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
