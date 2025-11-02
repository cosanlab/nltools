"""
Pytest configuration and fixtures for inference module tests.

Fixtures defined here are automatically available to all test files
in this directory and subdirectories.
"""

import pytest
import numpy as np

from nltools.backends import check_gpu_available


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    # 30 subjects, 100 features (small for fast tests)
    return np.random.randn(30, 100)


@pytest.fixture
def backends():
    """Return list of available backends."""
    backends_list = ["numpy"]
    if check_gpu_available()[0]:
        backends_list.append("torch")
    return backends_list
