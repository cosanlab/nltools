import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from nltools.simulator import Simulator
from nltools.data import Adjacency, Design_Matrix, Brain_Data
import os


@pytest.fixture(scope="module", params=["2mm"])
def sim_brain_data():
    np.random.seed(0)
    # MNI_Template["resolution"] = request.params
    sim = Simulator()
    sigma = 1
    y = [0, 1]
    n_reps = 3
    dat = sim.create_data(y, sigma, reps=n_reps)
    dat.X = pd.DataFrame(
        {"Intercept": np.ones(len(dat.Y)), "X1": np.array(dat.Y).flatten()}, index=None
    )
    return dat


@pytest.fixture(scope="module")
def sim_design_matrix():
    np.random.seed(0)
    # Design matrices are specified in terms of sampling frequency
    TR = 2.0
    sampling_freq = 1.0 / TR
    return Design_Matrix(
        np.random.randint(2, size=(500, 4)),
        columns=["face_A", "face_B", "house_A", "house_B"],
        sampling_freq=sampling_freq,
    )


@pytest.fixture(scope="module")
def sim_adjacency_single():
    np.random.seed(0)
    # Create a positive definite covariance matrix
    cov_matrix = np.array(
        [
            [1.0, 0.5, 0.1, 0.2],
            [0.5, 1.0, 0.3, 0.1],
            [0.1, 0.3, 1.0, 0.2],
            [0.2, 0.1, 0.2, 1.0],
        ]
    )
    sim = np.random.multivariate_normal([0, 0, 0, 0], cov_matrix, 100)
    data = pairwise_distances(sim.T, metric="correlation")
    labels = ["v_%s" % (x + 1) for x in range(sim.shape[1])]
    return Adjacency(data, labels=labels)


@pytest.fixture(scope="module")
def sim_adjacency_multiple():
    np.random.seed(0)
    n = 10
    # Create a positive definite covariance matrix
    cov_matrix = np.array(
        [
            [1.0, 0.5, 0.1, 0.2],
            [0.5, 1.0, 0.3, 0.1],
            [0.1, 0.3, 1.0, 0.2],
            [0.2, 0.1, 0.2, 1.0],
        ]
    )
    sim = np.random.multivariate_normal([0, 0, 0, 0], cov_matrix, 100)
    data = pairwise_distances(sim.T, metric="correlation")
    dat_all = []
    for t in range(n):
        tmp = data
        dat_all.append(tmp)
    labels = ["v_%s" % (x + 1) for x in range(sim.shape[1])]
    return Adjacency(dat_all, labels=labels)


@pytest.fixture(scope="module")
def sim_adjacency_directed():
    sim_directed = np.array(
        [
            [1, 0.5, 0.3, 0.4],
            [0.8, 1, 0.2, 0.1],
            [0.7, 0.6, 1, 0.5],
            [0.85, 0.4, 0.3, 1],
        ]
    )
    labels = ["v_%s" % (x + 1) for x in range(sim_directed.shape[1])]
    return Adjacency(sim_directed, matrix_type="directed", labels=labels)


@pytest.fixture(scope="module")
def old_h5_brain(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "old_brain.h5")


@pytest.fixture(scope="module")
def new_h5_brain(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "new_brain.h5")


@pytest.fixture(scope="module")
def old_h5_adj_single(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "old_single.h5")


@pytest.fixture(scope="module")
def new_h5_adj_single(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "new_single.h5")


@pytest.fixture(scope="module")
def old_h5_adj_double(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "old_double.h5")


@pytest.fixture(scope="module")
def new_h5_adj_double(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "new_double.h5")


@pytest.fixture(scope="module")
def regress_result(sim_brain_data):
    # Create labels based on actual data shape
    n_conditions = sim_brain_data.shape[0]
    labels = ["condition_" + str(i) for i in range(n_conditions)]
    # Make face and house special indices for testing
    if n_conditions >= 4:
        labels[3] = "face"
    if n_conditions >= 5:
        labels[4] = "house"

    # 64 "TRs"
    fake_timeseries = Brain_Data([sim_brain_data] * 8)

    return {
        "z_score": sim_brain_data,
        "t": sim_brain_data.copy(),
        "p": sim_brain_data.copy(),
        "beta": sim_brain_data.copy(),
        "se": sim_brain_data.copy(),
        "rsquared": sim_brain_data.copy()[0],  # 1 value per voxel
        "residual": fake_timeseries - fake_timeseries.mean(),
        "predicted": fake_timeseries,
        "labels": labels,
    }
