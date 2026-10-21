# Riemannian optimization on low-rank tensor train manifolds for the Gross–Pitaevskii equation

[![arXiv](https://img.shields.io/badge/arXiv-2609.37103-b31b1b.svg)](https://arxiv.org/abs/2609.37103)

Official repository for the paper [*Riemannian optimization on low-rank tensor train manifolds for the Gross–Pitaevskii equation*](https://arxiv.org/abs/2609.37103) by Ivan Bioli, Carlo Marcati and Maxim Rakhuba.

## Authors

- **Ivan Bioli**<sup>1,2</sup> — [ivan.bioli@unipv.it](mailto:ivan.bioli@unipv.it)
- **Carlo Marcati**<sup>3</sup> — [carlo.marcati@univ-lyon1.fr](mailto:carlo.marcati@univ-lyon1.fr)
- **Maxim Rakhuba**<sup>4</sup> — [mrakuba@hse.ru](mailto:mrakuba@hse.ru)

<sup>1</sup> Dipartimento di Matematica, Università di Pavia, 27100 Pavia, Italy<br>
<sup>2</sup> Dipartimento di Ingegneria Civile e Architettura, Università di Pavia, 27100 Pavia, Italy<br>
<sup>3</sup> Université Lyon 1, Centrale Lyon, INSA Lyon, Université Jean Monnet, CNRS, ICJ UMR5208, 69622 Villeurbanne, France<br>
<sup>4</sup> HSE University, Faculty of Computer Science, 109028 Moscow, Russian Federation

## Abstract

This work is concerned with the numerical computation of Gross–Pitaevskii ground states using the tensor train (TT) format. We regard the problem as the minimization of the Gross–Pitaevskii energy functional under a unit mass constraint, and propose a first-order Riemannian optimization scheme that preserves both the mass constraint and the low-rank representation throughout the optimization, by operating on the manifold of unit-norm functions of fixed TT-rank. On this manifold, we derive the energy-adaptive Riemannian gradient and show how to compute it efficiently in the TT format. Spatial discretization is performed using a spectral method with numerical integration, which allows the required computations to be carried out efficiently while preserving the low-rank format. The method is shown to substantially reduce computational time compared to full-rank computations, while maintaining accuracy for both single- and multicomponent Gross–Pitaevskii equations.

## Overview

The code minimizes the discretized Gross–Pitaevskii energy on the manifold of
unit-norm functions of fixed TT rank with Riemannian gradient descent (R-GD)
and Riemannian nonlinear conjugate gradient (R-NLCG), using the energy-adaptive
(a_u) Riemannian gradient with the preconditioners of the paper, and compares
it with the same methods on full tensors (a_u and H¹ Sobolev gradients). It is
written in JAX (GPU, float64) on top of forks of
[ttax](https://github.com/fasghq/ttax) (TT arithmetic and automatic
differentiation on the TT manifold), [pymanopt](https://github.com/pymanopt/pymanopt)
(optimizers) and [ttpy](https://github.com/oseledets/ttpy) (cross approximation
of the initial guess); the 1D spectral-element matrices and the high-order
energy are assembled with [FEniCSx](https://fenicsproject.org).

This repository contains the code to reproduce every figure and table of the
numerical experiments section of the paper (the experiment drivers and the
analysis that produces the figures and tables) and the per-iteration logs of
the runs reported in the paper.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{bioli2026riemannian,
  title         = {Riemannian optimization on low-rank tensor train manifolds for the {G}ross--{P}itaevskii equation},
  author        = {Bioli, Ivan and Marcati, Carlo and Rakhuba, Maxim},
  year          = {2026},
  eprint        = {2609.37103},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  doi           = {10.48550/arXiv.2609.37103},
  url           = {https://arxiv.org/abs/2609.37103}
}
```

Citation metadata is also available in [`CITATION.cff`](CITATION.cff).

## Installation

Requirements: Linux, an NVIDIA GPU with a CUDA 13 driver, conda, a Fortran
compiler (installed by the environment file) and, optionally, a LaTeX
installation (for the exact fonts of the figures).

```bash
git clone --recursive https://github.com/IvanBioli/TT_for_GPE.git
cd TT_for_GPE
conda env create -f environment.yml
conda activate ttgpe
pip install -e src/external/ttax
pip install -e src/external/pymanopt
pip install --no-build-isolation src/external/ttpy
unzip src/external/EXP_SUM/1_x_rel/Rel1_x_n.zip -d src/external/EXP_SUM/1_x_rel/Rel1_x_n
```

If you cloned without `--recursive`, run `git submodule update --init --recursive`
first. The environment pins the versions used for the paper (Python 3.11.13,
JAX 0.7.2 with CUDA 13, numpy 2.3.0, FEniCSx 0.9.0). Check the installation with

```bash
python -c "import jax; print(jax.devices())"   # should list a CUDA device
```

## Quickstart

```bash
python -m pytest src/test -q                                  # unit tests (~1 min on a GPU)
python src/run.py --smoke                                     # every configuration, small size, 5 iterations
python src/experiments/analysis/analysis_new.py --smoke       # figures and tables of the smoke runs
python src/experiments/analysis/analysis_new.py --source csv  # figures and tables from data/arxiv_v1
```

`--smoke` writes the results to `results_smoke/` and the small versions of the
figures and tables to `results_smoke/paper_figures/`. `--source csv` produces the
paper's figures and tables in `figures/arxiv_v1/` from the logs of the paper's
runs shipped in `data/arxiv_v1` (no GPU needed).

## Reproducing the paper

`src/run.py` runs the experiments behind each figure or table; results are
pickled to `results/` (or `--results_dir`, or `$TTGPE_RESULTS_DIR`) and existing
results are skipped. `python src/experiments/analysis/analysis_new.py` then
produces the figures and tables in `results/paper_figures/` (`--results_dir` to
read another results directory). In the code, `N` is the
paper's `n − 1` (`N=399` is `n=400`); all runs use `k=4` and float64.

| Paper item | Runs | Command | Outputs | Runtime |
|---|---|---|---|---|
| Fig. 'R-GD vs R-NLCG' (`fig:exp1_CGvsSD`): 3D harmonic, N=399, S+V | 10 | `python src/run.py --only exp1` | `exp1_CGvsSD_iters.pdf`, `exp1_CGvsSD_time.pdf` | 8.5 h |
| Tab. preconditioners (`tab:exp2_precond`): harmonic, N=399, R-NLCG | 12 (4 shared with exp1) | `python src/run.py --only exp2` | `exp2_precond_table.tex` | +2.8 h |
| Fig. full vs TT, harmonic (`fig:exp3_fullvsTT`): N=399, 799 | 8 (5 shared with exp1) | `python src/run.py --only exp3` | `exp3_harmonic_N{399,799}_k4_time_shared.pdf` | +0.04 h |
| Fig. lattice potential (`fig:exp4_lattice`): N=399, 799 | 8 | `python src/run.py --only exp4` | `exp4_lattice_N{399,799}_k4_time_shared.pdf` | 4.1 h |
| Fig. Anderson localization (`fig:exp5_anderson`): 2D, N=1599, 3199, 6399 | 21 | `python src/run.py --only exp5` | `exp5_anderson_k4_N{1599,3199,6399}_k4_time_shared.pdf` | 11.7 h |
| Fig. multicomponent (`fig:exp9_multicomponent`): 2D, p=3, N=1023, 2047 | 10 | `python src/run.py --only exp9` | `exp9_multicomponent_k4_N{1023,2047}_k4_time_shared.pdf` | 0.5 h |
| Tab. multicomponent R-GD vs R-NLCG (`tab:exp9_multicomponent`): N=2047 | 10 (5 shared with exp9) | `python src/run.py --only tab9` | `exp9_multicomponent_CGvsSD_table.tex` | +0.7 h |
| Reference energy E* of the multicomponent figure: N=4095 | 5 | `python src/run.py --only ref9` | (used by exp9) | 2.4 h |

`python src/run.py` runs all 70 distinct runs (about 30 h on an RTX 4090);
`--dry_run` prints the commands. Each run can also be started directly, e.g.

```bash
python src/experiments/paper/paperexp_harmonic.py --method tt --N 399 --k 4 --dim 3 --beta 1000 \
    --pymanopt_optimizer ConjugateGradient --precond S+V --tt_rank 10 --max_tt_rank_u2 10
```

The runtimes above are approximate wall times on an RTX 4090, including the
per-iteration evaluation of the high-order energy used for the plots; two runs
stop at the 4 h time limit (R-GD with the full H¹ gradient at N=399, and the
full H¹ run at N=6399 for the Anderson potential).

## Hardware and runtime

All runs of the paper used a single NVIDIA RTX 4090 (24 GB) in float64. The
full-format runs are dominated by the per-iteration evaluation of the
high-order energy used only for logging (about 131 s per iteration for the 3D
problems at N=399); the plotted times exclude logging.

Peak memory of R-NLCG runs with the S+V preconditioner, measured on an RTX 4090
(`XLA_PYTHON_CLIENT_PREALLOCATE=false`):

| Problem | Method | Peak GPU memory | Peak host memory |
|---|---|---|---|
| harmonic 3D, N=399 | TT, rank 10 | 0.6 GB | 2.2 GB |
| harmonic 3D, N=399 | full a_u | 9.7 GB | 25.7 GB |
| multicomponent 2D, N=1023 | TT, rank 10 | 0.6 GB | 3.1 GB |
| multicomponent 2D, N=1023 | full a_u | 0.5 GB | 2.5 GB |

The full-format 3D runs need about 26 GB of host memory.

## Data

`data/arxiv_v1/` contains the logs of the 70 runs of `src/run.py` reported in the
paper: for each run, a CSV file with the per-iteration log (`cost`,
`gradient_norm`, `step_size`, `time`, `alpha`, `log_time`, `lmd`, `E_exact`,
`cg_iters` where applicable) and a `run.json` summary (stopping criterion,
iterations, time, optimizer settings), in the folder layout written by the
experiment scripts. `src/experiments/analysis/export_logs.py` writes the same
format for new results.

## Differences from the paper text

The following details are not stated in, or differ from, the text of arXiv v1:

1. The Anderson runs stop at gradient norm 10⁻³ (`min_gradient_norm`), not 10⁻⁶.
2. The multicomponent runs start from a constant initial guess, not from the Thomas–Fermi approximation.
3. The TT multicomponent gradient (route II) solves its linear systems with the adaptive relative tolerance min(‖grad‖^θ, κ), κ = 0.01, θ = 1, not 10⁻¹⁰. The single-component and full-format solves stop at relative residual 10⁻¹⁰ (Euclidean norm) or 200 iterations.
4. Besides the gradient norm, runs stop when the step size falls below 10⁻¹⁰ or after 4 h. All multicomponent TT runs and the harmonic TT runs of rank 5 stopped on the step size; R-GD with the full H¹ gradient (harmonic, N=399, 108 iterations) and the full H¹ run of the Anderson problem at N=6399 (460 iterations) stopped on time.
5. R-NLCG uses pymanopt's adaptive line search (sufficient decrease 0.5, contraction 0.5, at most 10 backtracking steps); R-GD uses pymanopt's backtracking line search (sufficient decrease 10⁻⁴, initial step optimism 2, at most 25 steps).
6. The TT multicomponent preconditioner approximates (S + V₀ + I)⁻¹ ⊗ I_p, while the full-format one is (S + V₀)⁻¹.
7. The multicomponent potential is 2¹²(1 − f₁(x) f₂(y)) plus the trap: the wells (V = 0) occupy one quarter of the 64 × 64 squares, as in the code of Altmann et al., rather than a checkerboard with values {0, 2¹²}.
8. E* in the relative energy error is the minimum final energy over the R-NLCG runs of the same experiment family (harmonic: exp1 and exp3; lattice; Anderson); for the multicomponent problem it is the minimum over finer-grid runs at N=4095 (`python src/run.py --only ref9`), not "computed on a finer grid and with higher TT ranks" for every problem.
9. The Thomas–Fermi initial guess is u² = (μ − V)₊/(2β) with μ chosen by bisection for unit mass. The arXiv v1 single-component runs used (μ − V/2)₊/(2β) (the Thomas–Fermi profile for 2β); fresh runs therefore differ slightly in their trajectories and iteration counts.
10. The S+V preconditioner uses the per-dimension potentials V₀: harmonic xᵢ²; lattice xᵢ² + 40/3; Anderson 0.25 + 0.25/ε²; multicomponent trap(xᵢ) + 2¹¹(1 − fᵢ(xᵢ)).
11. The Anderson disorder is `jax.random.bernoulli(PRNGKey(1), (40, 40))` (48.9% of the squares at ε⁻²), independent of N. The potential used to build the initial guess differs from the one used in the optimization on about 0.8% of the nodes (square boundaries); this only affects the initial guess.

## Repository layout

```
src/run.py                          experiment driver (one item per figure/table)
src/core/                           energy functionals, Riemannian gradients, manifolds, TT utilities
src/experiments/experiment_run.py   command-line interface of the experiment scripts
src/experiments/experiment_setup.py problem setup, optimizers, logging, result files
src/experiments/paper/              one script per problem (harmonic, lattice, Anderson, multicomponent)
src/experiments/analysis/           figures/tables (analysis_new.py) and CSV export (export_logs.py)
src/test/                           unit tests
src/external/                       forks of pymanopt, ttpy, ttax and the EXP_SUM tables (submodules)
data/arxiv_v1/                      per-iteration logs of the paper's runs
docs/forks.md                       changes made in the forks and pinned versions
```

## Dependencies and forks

The code uses forks of pymanopt, ttpy and ttax with small changes (logging
hooks, deterministic cross approximation, delta-format gradients); the pinned
commits and every change are documented in [`docs/forks.md`](docs/forks.md).
The exponential-sum tables of the TT preconditioner come from
[EXP_SUM](https://gitlab.mis.mpg.de/scicomp/EXP_SUM) by W. Hackbusch, which is
third-party, not redistributed here and has to be cited separately
(W. Hackbusch, *Computation of best L∞ exponential sums for 1/x by Remez'
algorithm*, Comput. Vis. Sci. 20 (2019)).

## License

The code in this repository is released under the Apache License 2.0 (see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). The submodules keep their own
licenses (pymanopt: BSD 3-Clause; ttpy and ttax: MIT; EXP_SUM: no license).
