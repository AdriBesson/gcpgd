# New Perspectives on Generalized Finite Rate of Innovation
[Ecole Polytechnique Fédérale de Lausanne (EPFL)]: http://www.epfl.ch/
[E-Scopics]: https://www.e-scopics.com/
[Center for Imaging]: https://imaging.epfl.ch/

Adrien Besson<sup>1</sup> and Matthieu Simeoni<sup>2</sup>, February 2024

<sup>1</sup>[E-Scopics], France

<sup>2</sup>[Center for Imaging], [Ecole Polytechnique Fédérale de Lausanne (EPFL)], Switzerland

Code used to reproduce the results presented in the paper entitled *New Perspectives on Generalized Finite Rate of Innovation*, submitted to IEEE Signal Processing Letters

## Abstract
The generalized finite rate of innovation (GenFRI) framework aims at reconstructing finite-rate-of-innovation (FRI) signals measured through a noisy linear measurement model. GenFRI has been recently recast as a structured low-rank optimization problem and the Cadzow projected gradient descent (CPGD) algorithm has been suggested to solve it. While CPGD works well in practice, only local convergence guarantees have been established. We introduce a different view of GenFRI under the light of the regularization by denoising (RED) framework. We recast GenFRI as a RED optimization problem in which a solution lies in the fixed-point set of the Cadzow denoiser. We propose the generalized CPGD (GCPGD) algorithm, a variant of CPGD which comes with stronger convergence guarantees. We show through numerical simulations that GCPGD outperforms state-of-the-art GenFRI algorithms.

## Repository Structure
The repository has been restructured and simplified. The previous `pyoneer` library and individual reproduction scripts have been consolidated as follows:
* `gcpgd_lib/`: Consolidates all GCPGD algorithms, operators, signal definitions, and certificate checks.
  * `toeplitz.py`: Toeplitz weights, Toeplitz projection, rank projection, and Cadzow denoiser.
  * `signal.py`: FRI location sampling, Fourier coefficients, and measurement operators.
  * `algorithm.py`: GCPGD iteration (using Gamma-gradient step), CPGD, and GenFRI algorithms.
  * `certificates.py`: Nontangentiality certificate, restricted/global mu calculation, and Lemma 4 bounds.
  * `recovery.py`: Annihilating-filter location recovery and circular/average match error calculation.
  * `stats.py`: Wilson binomial interval utility.
* `plots/`: Contains custom plotting styles and utilities.
* `reproduce_all_experiments.py`: A unified driver script to run all paper experiments under a clean CLI interface.

## Requirements
* Python environment (Tested on 3.10)
* Python packages: `numpy`, `scipy`, `matplotlib`

## Getting the code

You can download a copy of all the files in this repository by cloning the [git](https://git-scm.com/) repository:

```bash
git clone https://github.com/AdriBesson/gcpgd.git
```

or [download a zip archive](https://github.com/AdriBesson/gcpgd/archive/refs/heads/main.zip).

## Usage
The `reproduce_all_experiments.py` script is the entry point for reproducing all experiments described in the paper. It supports several subcommands depending on which claim or figure you wish to validate.

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

### Example
To reproduce the reconstruction accuracy simulation in Section IV.B with full fidelity and save the output plots to a custom folder:
```bash
python reproduce_all_experiments.py simulation --full --outdir ./plots_output
```

To run a fast validation of the inner alternating projection rates:
```bash
python reproduce_all_experiments.py rate --fast
```

## Contact
Adrien Besson (adribesson@gmail.com)

## License
Please cite the following paper when using the code:
A. Besson and M. Siméoni, "New Perspectives on Generalized Finite Rate of Innovation", submitted to IEEE Signal Processing Letters, 2024.

```
MIT License

Copyright (c) 2020 Matthieu SIMEONI, Adrien BESSON, Paul HURLEY and Martin VETTERLI

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
