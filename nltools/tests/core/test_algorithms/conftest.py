"""
Pytest configuration and fixtures for stats module tests.

Fixtures defined here are automatically available to all test files
in this directory and subdirectories.
"""

import pytest
import numpy as np
import polars as pl


@pytest.fixture
def random_state():
    """Provide a fixed random state for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def correlated_samples():
    """Generate two correlated 1D samples for permutation/correlation tests."""
    cov_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
    np.random.seed(42)
    dat = np.random.multivariate_normal([2, 6], cov_matrix, 1000)
    return dat[:, 0], dat[:, 1]


@pytest.fixture
def multisubject_correlated_data():
    """Generate correlated multi-subject data for ISC tests."""
    np.random.seed(42)
    return np.random.multivariate_normal(
        [0, 0, 0, 0, 0],
        [
            [1, 0.2, 0.5, 0.7, 0.3],
            [0.2, 1, 0.6, 0.1, 0.2],
            [0.5, 0.6, 1, 0.3, 0.1],
            [0.7, 0.1, 0.3, 1, 0.4],
            [0.3, 0.2, 0.1, 0.4, 1],
        ],
        500,
    )


@pytest.fixture
def outlier_data():
    """DataFrame with known outlier values for winsorize/trim tests."""
    return pl.DataFrame(
        {
            "x": [
                92,
                19,
                101,
                58,
                1053,
                91,
                26,
                78,
                10,
                13,
                -40,
                101,
                86,
                85,
                15,
                89,
                89,
                28,
                -5,
                41,
            ]
        }
    )


@pytest.fixture
def sub_roi_data():
    """Generate multi-subject ROI timeseries data for ISFC tests."""
    np.random.seed(42)
    sub_dat = []
    for _ in range(10):
        sub_dat.append(
            np.random.multivariate_normal(
                [0, 0, 0, 0, 0],
                [
                    [1, 0.2, 0.5, 0.7, 0.3],
                    [0.2, 1, 0.6, 0.1, 0.2],
                    [0.5, 0.6, 1, 0.3, 0.1],
                    [0.7, 0.1, 0.3, 1, 0.4],
                    [0.3, 0.2, 0.1, 0.4, 1],
                ],
                500,
            )
        )
    return sub_dat
