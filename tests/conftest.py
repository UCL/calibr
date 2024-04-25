"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

DEFAULT_SEED = 11741606243044876999


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--seed",
        type=int,
        nargs="*",
        default=[DEFAULT_SEED],
        help="Seed(s) for random number generators in tests",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "seed" in metafunc.fixturenames:
        metafunc.parametrize("seed", metafunc.config.getoption("seed"))


@pytest.fixture()
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
