# New Perspectives on Generalized Finite Rate of Innovation
[Ecole Polytechnique Fédérale de Lausanne (EPFL)]: http://www.epfl.ch/
[E-Scopics]: https://www.e-scopics.com/
[Center for Imaging]: https://imaging.epfl.ch/

Adrien Besson<sup>1</sup> and Matthieu Simeoni<sup>2</sup>
February 2024

<sup>1</sup>[E-Scopics], France

<sup>2</sup>[Center for Imaging], [Ecole Polytechnique Fédérale de Lausanne (EPFL)], Switzerland

Code used to reproduce the results presented in the paper entitled *New Perspectives on Generalized Finite Rate of Innovation*, submitted to IEEE Signal Processing Letters

## Abstract
The generalized finite rate of innovation~(GenFRI) framework aims at reconstructing finite-rate-of-innovation~(FRI) signals measured through a noisy linear measurement model. GenFRI has been recently recast as a structured low-rank optimization problem and the Cadzow projected gradient descent~(CPGD) algorithm has been suggested to solve it. While CPGD works well in practice, only local convergence guarantees have been established.We introduce a different view of GenFRI under the light of the regularization by denoising~(RED) framework. We recast GenFRI as a RED optimization problem in which a solution lies in the fixed-point set of the Cadzow denoiser. We propose the generalized CPGD~(GCPGD) algorithm, a variant of CPGD which comes with stronger convergence guarantees. We show through numerical simulations that GCPGD outperforms state-of-the-art GenFRI algorithms. 

## Requirements
  * Python environment (Tested on 3.10)
  * Python packages: numpy, scipy, matplotlib, joblib

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/AdriBesson/gcpgd.git

or [download a zip archive](https://github.com/AdriBesson/gcpgd/archive/refs/heads/main.zip).

## Usage
1. If you want to reproduce the experiments, run one of the following scripts
  * `reproduce_lipschitz_cadzow.py` reproduces the results of the Section IV.A entitled "Lipschitz Constant of Cadzow Denoising"
  * `reproduce_simulation_results.py` reproduces the results of the Section IV.B entitled "Reconstruction Accuracy of GCPGD"


## Contact
 Adrien Besson (adribesson@gmail.com)

## License
[License](LICENSE.txt) for non-commercial use of the software. Please cite the following paper when using the code:

A. Besson and M. Siméoni, "New Perspectives on Generalized Finite Rate of Innovation", submitted to IEEE Signal Processing Letters, 2024.
