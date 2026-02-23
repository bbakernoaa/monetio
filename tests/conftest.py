import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def is_ci():
    return os.environ.get("CI", "false").lower() in {"true", "yes", "1", ""}


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"


def pytest_configure(config):
    config.addinivalue_line("markers", "network: mark test as requiring network access")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    is_ci = os.environ.get("CI", "false").lower() in {"true", "yes", "1", ""}
    if "network" in item.keywords and is_ci:
        pytest.skip("Skip network test in CI")
