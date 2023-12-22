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

Parallelised Bayesian calibration of simulations using Gaussian process emulation.

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
