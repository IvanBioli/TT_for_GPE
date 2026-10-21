# External dependencies

This directory contains the third-party code used by the project, included as
git submodules pinned by commit:

- `pymanopt/` — fork of [pymanopt](https://github.com/pymanopt/pymanopt) (optimizers, logging hooks)
- `ttpy/` — fork of the legacy 1.x line of [ttpy](https://github.com/oseledets/ttpy) (cross approximation)
- `ttax/` — fork of [ttax](https://github.com/fasghq/ttax) (TT arithmetic and automatic differentiation in JAX)
- `EXP_SUM/` — [exponential-sum tables](https://gitlab.mis.mpg.de/scicomp/EXP_SUM) for the TT preconditioner

Initialize them with `git submodule update --init --recursive` and install them
as described in the main README ("Installation"). The changes made in the forks
and the pinned commits are documented in `docs/forks.md`.
