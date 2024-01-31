# calibr

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![License][license-badge]](./LICENSE.md)

<!--
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
-->

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/UCL/calibr/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/UCL/calibr/actions/workflows/tests.yml
[linting-badge]:            https://github.com/UCL/calibr/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/UCL/calibr/actions/workflows/linting.yml
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/calibr
[conda-link]:               https://github.com/conda-forge/calibr-feedstock
[pypi-link]:                https://pypi.org/project/calibr/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/calibr
[pypi-version]:             https://img.shields.io/pypi/v/calibr
[license-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

Parallelized Bayesian calibration of simulations using Gaussian process emulation.

`calibr` is a Python implementation of the algorithm described in _Parallel Gaussian
process surrogate Bayesian inference with noisy likelihood evaluations_ (Järvenpää,
Gutmann, Vehtari and Marttinen; 2021)
([doi:10.1214/20-BA1200](https://doi.org/10.1214/20-BA1200),
[arxiv:1905.01252](https://arxiv.org/abs/1905.01252)). It is designed to allow
estimation of the posterior distribution on the unknown parameters of expensive to
evaluate simulator models given observed data, using a batch sequential design strategy
which iterates fitting a Gaussian process emulator to a set of evaluations of the
(unnormalized) posterior density for the model and using the emulator to identify a new
batch of model parameters at which to evaluate the posterior density which minimize a
measure of the expected uncertainty in the emulation of the posterior density.

The posterior density can be evaluated at the parameter values in each batch in
parallel, providing the opportunity for speeding up calibration runs on multi-core and
multi-node high performance computing systems. The acquisition functions used to choose
new parameter values to evaluate are implemented using the high-performance numerical
computing framework [JAX](https://jax.readthedocs.io/en/latest/), with the
gradient-based optimization of these acquisition functions exploiting JAX's support for
automatic differentiation.

The package is still in the early stages of development, with only a subset of the
algorithmic variants proposed by Järvenpää, Gutmann, Vehtari and Marttinen (2021)
currently implemented. In particular there is no support yet for models with noisy
likelihood evaluations or greedy strategies for optimizing the acquisition functions.
Expect lots of rough edges!

This project is developed in collaboration with the [Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London.

## About

### Project Team

- Matt Graham ([matt-graham](https://github.com/matt-graham))

### Research Software Engineering Contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

## Getting Started

### Prerequisites

`calibr` requires Python 3.10&ndash;3.12.

### Installation

We recommend installing in a project specific virtual environment created using a environment management tool such as [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) or [Conda](https://conda.io/projects/conda/en/latest/). To install the latest development version of `calibr` using `pip` in the currently active environment run

```sh
pip install git+https://github.com/UCL/calibr.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/UCL/calibr.git
```

and then install in editable mode by running

```sh
pip install -e .
```

### Running locally

How to run the application on your local system.

### Running tests

Tests can be run across all compatible Python versions in isolated environments using
[`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

## Acknowledgements

This work was funded by a grant from the ExCALIBUR programme.
