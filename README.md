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

## Code availability

The code to reproduce the results of the paper will be made publicly available in this repository upon publication. For questions about the code, please contact [ivan.bioli@unipv.it](mailto:ivan.bioli@unipv.it).

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
