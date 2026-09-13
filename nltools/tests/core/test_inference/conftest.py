"""
Pytest configuration and fixtures for inference module tests.

Fixtures defined here are automatically available to all test files
in this directory and subdirectories.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    # 30 subjects, 100 features (small for fast tests)
    return np.random.randn(30, 100)
