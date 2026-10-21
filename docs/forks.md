# Third-party code: forks and pinned versions

The code depends on three forked libraries and on one data repository, all
included as git submodules under `src/external/` and pinned by commit. The
forks carry small changes that the experiments rely on; this page lists every
change, why it was made and where it is used.

| Submodule | Fork | Pinned commit | Branch | Upstream | License |
|---|---|---|---|---|---|
| `src/external/pymanopt` | https://github.com/IvanBioli/pymanopt | `4fe1ddaad0b1136a6db02592b4fb59a2a8b3840d` | `ttgpe-v1` | https://github.com/pymanopt/pymanopt | BSD 3-Clause |
| `src/external/ttpy` | https://github.com/IvanBioli/ttpy | `f0f11dbe2f2ceea0c4fd44fd85f007af7a9c97e6` | `master` | https://github.com/oseledets/ttpy (legacy 1.x line) | MIT |
| `src/external/ttax` | https://github.com/IvanBioli/ttax | `73ddc7d291db056634567e1daee6cd4ec5eeb763` | `develop` | https://github.com/fasghq/ttax | MIT |
| `src/external/EXP_SUM` | (not a fork) https://gitlab.mis.mpg.de/scicomp/EXP_SUM | `7ee4982e8ff178b5a0b00c09c435c764333fa86f` | `main` | — | none (see below) |

Installation (see also the main README): pymanopt and ttax are installed in
editable mode, ttpy is built from source without build isolation (it needs
gfortran and numpy.distutils, hence Python ≤ 3.11):

```bash
pip install -e src/external/ttax
pip install -e src/external/pymanopt
pip install --no-build-isolation src/external/ttpy
unzip src/external/EXP_SUM/1_x_rel/Rel1_x_n.zip -d src/external/EXP_SUM/1_x_rel/Rel1_x_n
```

## pymanopt

Fork of pymanopt 2.2.1 (merge base `a1f52e7`). The fork adds logging hooks
used to record the high-order energy, the CG iteration counts and the timings
at every iteration of the optimizers.

| Commit | Change | Why | Used in this repository |
|---|---|---|---|
| `c3d2482` | `log_callable` argument of the optimizers: a user function whose returned dict is added to every log entry | log quantities that are not part of the optimizer state (high-order energy `E_exact`, Lagrange multiplier, inner CG iterations) | `experiments/experiment_setup.py` (`log_callable`, `log_callable_full`) |
| `1ec2b43` | `ConstantLineSearcher` (fixed step size) | experimentation | not used by the paper's runs |
| `5ad6ece` | log iteration 0 and the keys `step_size`, `time`, `log_time` (time spent in logging) for `SteepestDescent` and `ConjugateGradient`; also adds an unused "SteepestDescent" β rule | convergence plots versus time that exclude logging overhead | every run; `analysis_new.compute_time` |
| `2276726` | stop when the gradient norm is NaN | stop runs that break down instead of iterating until the time limit | every run (stopping criterion) |
| `4e56d1b` | do not force `jax_enable_x64` in the JAX backend | the precision is set by the experiment scripts | every run |
| `4fe1dda` | `ConjugateGradient`: drop the references to the previous iterate and gradient and call `gc.collect()` once per iteration | keeps GPU memory bounded in long runs with large JAX arrays | every R-NLCG run |

Upstream status: none of these changes exists upstream (master has since
moved to a new backend API). Three pull requests were prepared against
upstream `master` (branches of the fork):

| Branch | Change | Upstream issue |
|---|---|---|
| `pr/logging-hooks` | `log_callable` hook for every optimizer; R-GD and R-NLCG log iteration 0 and the final iterate (re-implemented on the current `Optimizer._add_log_entry`) | pymanopt#297 |
| `pr/nan-gradnorm-stop` | stop when the gradient norm is NaN | — |
| `pr/constant-line-search` | `ConstantLineSearcher` | pymanopt#133 |

The forced-x64 opt-out and the `gc.collect()` change are specific to this
project and were not proposed. Links to the pull requests: _to be added once
opened_.

Compatibility with current upstream: the fork commits (including the gc change,
with the x64 opt-out re-implemented in the new `backends/jax_backend.py`) were
rebased onto upstream `master` (`e97efe3`, October 2026) in the fork branch
`compat/upstream-master`. With it, this repository's tests pass and short runs
of every configuration give results identical to the pinned fork. The pin stays
on `ttgpe-v1`, which is the version used for the paper.

## ttpy

Fork of the legacy ttpy 1.x code base (upstream tag `ttpy1`, commit `22dff3d`).
ttpy is used to build the Thomas–Fermi and constant initial guesses in TT
format by cross approximation (`core/tt_utils.tt_from_fun`, which calls
`tt.multifuncrs2`, `tt.vector.from_list`/`to_list` and `round`).

| Commit | Change | Why |
|---|---|---|
| `dff1ddc` | replace the removed `np.object` | compatibility with numpy ≥ 1.24 |
| `b26e599` | `rng` argument of `multifuncrs2` (intermediate commit, superseded by the next one) | reproducible cross approximation |
| `f0f11db` | deterministic `multifuncrs2`: all random draws go through the given `rng` | reproducible initial guesses |

Not proposed upstream: ttpy's `master` branch is a different code base (the
"ttpy2" rewrite), to which these patches do not apply.

## ttax

Fork of ttax (fast-forward of upstream `develop` at `6e18887`, 2021). Upstream
has not been updated since 2021, so the fork is kept and pinned rather than
proposed upstream.

| Commit | Change | Used in this repository |
|---|---|---|
| `854a364` | `.gitignore` | — |
| `17eab10` | `ttax.grad`/`ttax.project` return the tangent vector in delta format (no conversion back to a TT tensor) — a breaking API change | the TT Riemannian gradients and projections (`core/energy.py`, `core/fixed_tt_rank.py`) rely on the delta format |
| `037cba8` | `fuse(..., argnums=...)` | `core/energy.py` (`fuse` decorator), `experiments/experiment_setup.py` |
| `7859613` | Hessian-vector product with input and output in delta format | not used by the paper's runs |
| `73ddc7d` | fix of the einsum index letters for high-order cores | used implicitly by `fuse` |

## EXP_SUM

Tables of exponential-sum approximations of 1/x by W. Hackbusch (see
W. Hackbusch, *Computation of best L∞ exponential sums for 1/x by Remez'
algorithm*, Comput. Vis. Sci. 20 (2019)), used to build the TT preconditioner
(`core/exp_sum.py`, files `1_x_rel/Rel1_x_n/Rel1_x_n{k}.{a}E{b}`; the paper's
runs use k = 10 terms). The repository has no license, so its content is not
redistributed here: the submodule only points to the upstream commit, and the
tables are unzipped locally after cloning.
