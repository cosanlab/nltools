import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from nltools.simulator import Simulator
from nltools.data import Adjacency, Design_Matrix
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
    test_dir = os.path.dirname(request.module.__file__)
    return os.path.join(test_dir, "old_brain.h5")


@pytest.fixture(scope="module")
def new_h5_brain(request):
    test_dir = os.path.dirname(request.module.__file__)
    return os.path.join(test_dir, "new_brain.h5")


@pytest.fixture(scope="module")
def old_h5_adj_single(request):
    test_dir = os.path.dirname(request.module.__file__)
    return os.path.join(test_dir, "old_single.h5")


@pytest.fixture(scope="module")
def new_h5_adj_single(request):
    test_dir = os.path.dirname(request.module.__file__)
    return os.path.join(test_dir, "new_single.h5")


@pytest.fixture(scope="module")
def old_h5_adj_double(request):
    test_dir = os.path.dirname(request.module.__file__)
    return os.path.join(test_dir, "old_double.h5")


@pytest.fixture(scope="module")
def new_h5_adj_double(request):
    test_dir = os.path.dirname(request.module.__file__)
    return os.path.join(test_dir, "new_double.h5")
