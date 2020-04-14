import os
import pytest
import numpy as np
import nibabel as nb
import pandas as pd
from nltools.simulator import Simulator
from nltools.data import Brain_Data, Adjacency, Groupby
from nltools.stats import threshold, align
from nltools.mask import create_sphere, roi_to_brain
from nltools.utils import get_resource_path
from nltools.mask import expand_mask
# from nltools.prefs import MNI_Template


shape_3d = (91, 109, 91)
shape_2d = (6, 238955)


def test_load(tmpdir):
    sim = Simulator()
    sigma = 1
    y = [0, 1]
    n_reps = 3
    output_dir = str(tmpdir)
    dat = sim.create_data(y, sigma, reps=n_reps, output_dir=output_dir)

    # if MNI_Template["resolution"] == '2mm':
    #     shape_3d = (91, 109, 91)
    #     shape_2d = (6, 238955)
    # elif MNI_Template["resolution"] == '3mm':
    #     shape_3d = (60, 72, 60)
    #     shape_2d = (6, 71020)

    y = pd.read_csv(
        os.path.join(str(tmpdir.join("y.csv"))), header=None, index_col=None
    )
    # holdout = pd.read_csv(os.path.join(str(tmpdir.join('rep_id.csv'))), header=None, index_col=None)

    # Test load list of 4D images
    file_list = [str(tmpdir.join("data.nii.gz")), str(tmpdir.join("data.nii.gz"))]
    dat = Brain_Data(file_list)
    dat = Brain_Data([nb.load(x) for x in file_list])

    # Test load list
    dat = Brain_Data(data=str(tmpdir.join("data.nii.gz")), Y=y)

    # Test Write
    dat.write(os.path.join(str(tmpdir.join("test_write.nii"))))
    assert Brain_Data(os.path.join(str(tmpdir.join("test_write.nii"))))

    # Test i/o for hdf5
    dat.write(os.path.join(str(tmpdir.join("test_write.h5"))))
    b = Brain_Data(os.path.join(tmpdir.join("test_write.h5")))
    for k in ["X", "Y", "mask", "nifti_masker", "file_name", "data"]:
        if k == "data":
            assert np.allclose(b.__dict__[k], dat.__dict__[k])
        elif k in ["X", "Y"]:
            assert all(b.__dict__[k].eq(dat.__dict__[k]).values)
        elif k == "mask":
            assert np.allclose(b.__dict__[k].affine, dat.__dict__[k].affine)
            assert np.allclose(b.__dict__[k].get_data(), dat.__dict__[k].get_data())
            assert b.__dict__[k].get_filename() == dat.__dict__[k].get_filename()
        elif k == "nifti_masker":
            assert np.allclose(b.__dict__[k].affine_, dat.__dict__[k].affine_)
            assert np.allclose(
                b.__dict__[k].mask_img.get_data(), dat.__dict__[k].mask_img.get_data()
            )
        else:
            assert b.__dict__[k] == dat.__dict__[k]


def test_shape(sim_brain_data):
    assert sim_brain_data.shape() == shape_2d

def test_mean(sim_brain_data):
    assert sim_brain_data.mean().shape()[0] == shape_2d[1]
    assert sim_brain_data.mean().shape()[0] == shape_2d[1]
    assert len(sim_brain_data.mean(axis=1)) == shape_2d[0]
    with pytest.raises(ValueError):
        sim_brain_data.mean(axis='1')
    assert isinstance(sim_brain_data[0].mean(), (float, np.floating))

def test_median(sim_brain_data):
    assert sim_brain_data.median().shape()[0] == shape_2d[1]
    assert sim_brain_data.median().shape()[0] == shape_2d[1]
    assert len(sim_brain_data.median(axis=1)) == shape_2d[0]
    with pytest.raises(ValueError):
        sim_brain_data.median(axis='1')
    assert isinstance(sim_brain_data[0].median(), (float, np.floating))

def test_std(sim_brain_data):
    assert sim_brain_data.std().shape()[0] == shape_2d[1]

def test_sum(sim_brain_data):
    s = sim_brain_data.sum()
    assert s.shape() == sim_brain_data[1].shape()

def test_add(sim_brain_data):
    new = sim_brain_data + sim_brain_data
    assert new.shape() == shape_2d
    value = 10
    assert (value + sim_brain_data[0]).mean() == (sim_brain_data[0] + value).mean()


def test_subtract(sim_brain_data):
    new = sim_brain_data - sim_brain_data
    assert new.shape() == shape_2d
    value = 10
    assert (-value - (-1) * sim_brain_data[0]).mean() == (
        sim_brain_data[0] - value
    ).mean()


def test_multiply(sim_brain_data):
    new = sim_brain_data * sim_brain_data
    assert new.shape() == shape_2d
    value = 10
    assert(value * sim_brain_data[0]).mean() == (sim_brain_data[0] * value).mean()
    c1 = [.5, .5, -.5, -.5]
    new = sim_brain_data[0:4]*c1
    new2 = sim_brain_data[0]*.5 + sim_brain_data[1]*.5 - sim_brain_data[2]*.5 - sim_brain_data[3]*.5
    np.testing.assert_almost_equal((new-new2).sum(), 0, decimal=4)

def test_divide(sim_brain_data):
    new = sim_brain_data / sim_brain_data
    assert new.shape() == shape_2d
    np.testing.assert_almost_equal(new.mean(axis=0).mean(), 1, decimal=6)
    value = 10
    new2 = sim_brain_data/value
    np.testing.assert_almost_equal(((new2*value) - new2).mean().mean(), 0, decimal=2)
    
def test_indexing(sim_brain_data):
    index = [0, 3, 1]
    assert len(sim_brain_data[index]) == len(index)
    index = range(4)
    assert len(sim_brain_data[index]) == len(index)
    index = sim_brain_data.Y == 1
    assert len(sim_brain_data[index.values.flatten()]) == index.values.sum()
    assert len(sim_brain_data[index]) == index.values.sum()
    assert len(sim_brain_data[:3]) == 3
    d = sim_brain_data.to_nifti()
    assert d.shape[0:3] == shape_3d
    assert Brain_Data(d)

def test_concatenate(sim_brain_data):
    out = Brain_Data([x for x in sim_brain_data])
    assert isinstance(out, Brain_Data)
    assert len(out) == len(sim_brain_data)

def test_append(sim_brain_data):
    assert sim_brain_data.append(sim_brain_data).shape()[0] == shape_2d[0] * 2

def test_ttest(sim_brain_data):
    out = sim_brain_data.ttest()
    assert out['t'].shape()[0] == shape_2d[1]

def test_distance(sim_brain_data):
    distance = sim_brain_data.distance(metric='correlation')
    assert isinstance(distance, Adjacency)
    assert distance.square_shape()[0] == shape_2d[0]

def test_regress(sim_brain_data):
    sim_brain_data.X = pd.DataFrame(
        {
            "Intercept": np.ones(len(sim_brain_data.Y)),
            "X1": np.array(sim_brain_data.Y).flatten(),
        },
        index=None,
    )
    # OLS
    out = sim_brain_data.regress()
    assert type(out["beta"].data) == np.ndarray
    assert type(out["t"].data) == np.ndarray
    assert type(out["p"].data) == np.ndarray
    assert type(out["residual"].data) == np.ndarray
    assert out["beta"].shape() == (2, shape_2d[1])
    assert out["t"][1].shape()[0] == shape_2d[1]

    # Robust OLS
    out = sim_brain_data.regress(mode="robust")
    assert type(out["beta"].data) == np.ndarray
    assert type(out["t"].data) == np.ndarray
    assert type(out["p"].data) == np.ndarray
    assert type(out["residual"].data) == np.ndarray
    assert out["beta"].shape() == (2, shape_2d[1])
    assert out["t"][1].shape()[0] == shape_2d[1]

    # Test threshold
    i = 1
    tt = threshold(out["t"][i], out["p"][i], 0.05)
    assert isinstance(tt, Brain_Data)

def test_randomise(sim_brain_data):
    sim_brain_data.X = pd.DataFrame({"Intercept": np.ones(len(sim_brain_data.Y))})

    out = sim_brain_data.randomise(n_permute=10)
    assert type(out["beta"].data) == np.ndarray
    assert type(out["t"].data) == np.ndarray
    assert type(out["p"].data) == np.ndarray
    assert out["beta"].shape() == (shape_2d[1],)
    assert out["t"].shape() == (shape_2d[1],)

    sim_brain_data.X = pd.DataFrame(
        {
            "Intercept": np.ones(len(sim_brain_data.Y)),
            "X1": np.random.randn(len(sim_brain_data.Y)),
        }
    )

    out = sim_brain_data.randomise(n_permute=10)
    assert type(out["beta"].data) == np.ndarray
    assert type(out["t"].data) == np.ndarray
    assert type(out["p"].data) == np.ndarray
    assert out["beta"].shape() == (2, shape_2d[1],)
    assert out["t"].shape() == (2, shape_2d[1],)


def test_apply_mask(sim_brain_data):
    s1 = create_sphere([12, 10, -8], radius=10)
    assert isinstance(s1, nb.Nifti1Image)
    masked_dat = sim_brain_data.apply_mask(s1)
    assert masked_dat.shape()[1] == np.sum(s1.get_data() != 0)
    masked_dat = sim_brain_data.apply_mask(s1, resample_mask_to_brain=True)
    assert masked_dat.shape()[1] == np.sum(s1.get_data() != 0)


def test_extract_roi(sim_brain_data):
    mask = create_sphere([12, 10, -8], radius=10)
    assert len(sim_brain_data.extract_roi(mask, metric='mean')) == shape_2d[0]
    assert len(sim_brain_data.extract_roi(mask, metric='median')) == shape_2d[0]
    n_components = 2
    assert sim_brain_data.extract_roi(mask, metric='pca', n_components=n_components).shape == (n_components, shape_2d[0])
    with pytest.raises(NotImplementedError):
        sim_brain_data.extract_roi(mask, metric='p')

    assert isinstance(sim_brain_data[0].extract_roi(mask, metric='mean'), (float, np.floating))
    assert isinstance(sim_brain_data[0].extract_roi(mask, metric='median'), (float, np.floating))
    with pytest.raises(ValueError):
        sim_brain_data[0].extract_roi(mask, metric='pca')
    with pytest.raises(NotImplementedError):
        sim_brain_data[0].extract_roi(mask, metric='p')

    s1 = create_sphere([15, 10, -8], radius=10)
    s2 = create_sphere([-15, 10, -8], radius=10)
    s3 = create_sphere([0, -15, -8], radius=10)
    masks = Brain_Data([s1, s2, s3])
    mask = roi_to_brain([1,2,3], masks)
    assert len(sim_brain_data[0].extract_roi(mask, metric='mean')) == len(masks)
    assert len(sim_brain_data[0].extract_roi(mask, metric='median')) == len(masks)
    assert sim_brain_data.extract_roi(mask, metric='mean').shape == (len(masks), shape_2d[0])
    assert sim_brain_data.extract_roi(mask, metric='median').shape == (len(masks), shape_2d[0])
    assert len(sim_brain_data.extract_roi(mask, metric='pca', n_components=n_components)) == len(masks)

def test_r_to_z(sim_brain_data):
    z = sim_brain_data.r_to_z()
    assert z.shape() == sim_brain_data.shape()

def test_copy(sim_brain_data):
    d_copy = sim_brain_data.copy()
    assert d_copy.shape() == sim_brain_data.shape()

def test_detrend(sim_brain_data):
    detrend = sim_brain_data.detrend()
    assert detrend.shape() == sim_brain_data.shape()

def test_standardize(sim_brain_data):
    s = sim_brain_data.standardize()
    assert s.shape() == sim_brain_data.shape()
    assert np.isclose(np.sum(s.mean().data), 0, atol=0.1)
    s = sim_brain_data.standardize(method="zscore")
    assert s.shape() == sim_brain_data.shape()
    assert np.isclose(np.sum(s.mean().data), 0, atol=0.1)


def test_smooth(sim_brain_data):
    smoothed = sim_brain_data.smooth(5.0)
    assert isinstance(smoothed, Brain_Data)
    assert smoothed.shape() == sim_brain_data.shape()


def test_groupby_aggregate(sim_brain_data):
    s1 = create_sphere([12, 10, -8], radius=10)
    s2 = create_sphere([22, -2, -22], radius=10)
    mask = Brain_Data([s1, s2])
    d = sim_brain_data.groupby(mask)
    assert isinstance(d, Groupby)
    mn = sim_brain_data.aggregate(mask, "mean")
    assert isinstance(mn, Brain_Data)
    assert len(mn.shape()) == 1

def test_threshold():
    s1 = create_sphere([12, 10, -8], radius=10)
    s2 = create_sphere([22, -2, -22], radius=10)
    mask = Brain_Data(s1) * 5
    mask = mask + Brain_Data(s2)

    m1 = mask.threshold(upper=0.5)
    m2 = mask.threshold(upper=3)
    m3 = mask.threshold(upper="98%")
    m4 = Brain_Data(s1) * 5 + Brain_Data(s2) * -0.5
    m4 = mask.threshold(upper=0.5, lower=-0.3)
    assert np.sum(m1.data > 0) > np.sum(m2.data > 0)
    assert np.sum(m1.data > 0) == np.sum(m3.data > 0)
    assert np.sum(m4.data[(m4.data > -0.3) & (m4.data < 0.5)]) == 0
    assert np.sum(m4.data[(m4.data < -0.3) | (m4.data > 0.5)]) > 0

    # Test Regions
    r = mask.regions(min_region_size=10)
    m1 = Brain_Data(s1)
    m2 = r.threshold(1, binarize=True)
    assert len(np.unique(r.to_nifti().get_data())) == 2
    diff = m2 - m1
    assert np.sum(diff.data) == 0


def test_bootstrap(sim_brain_data):
    masked = sim_brain_data.apply_mask(create_sphere(radius=10, coordinates=[0, 0, 0]))
    n_samples = 3
    b = masked.bootstrap("mean", n_samples=n_samples)
    assert isinstance(b["Z"], Brain_Data)
    b = masked.bootstrap("std", n_samples=n_samples)
    assert isinstance(b["Z"], Brain_Data)
    b = masked.bootstrap("predict", n_samples=n_samples, plot=False)
    assert isinstance(b["Z"], Brain_Data)
    b = masked.bootstrap(
        "predict",
        n_samples=n_samples,
        plot=False,
        cv_dict={"type": "kfolds", "n_folds": 3},
    )
    assert isinstance(b["Z"], Brain_Data)
    b = masked.bootstrap("predict", n_samples=n_samples, save_weights=True, plot=False)
    assert len(b["samples"]) == n_samples


def test_predict(sim_brain_data):
    holdout = np.array([[x] * 2 for x in range(3)]).flatten()
    stats = sim_brain_data.predict(
        algorithm="svm",
        cv_dict={"type": "kfolds", "n_folds": 2},
        plot=False,
        **{"kernel": "linear"}
    )

    # Support Vector Regression, with 5 fold cross-validation with Platt Scaling
    # This will output probabilities of each class
    stats = sim_brain_data.predict(
        algorithm="svm",
        cv_dict=None,
        plot=False,
        **{"kernel": "linear", "probability": True}
    )
    assert isinstance(stats["weight_map"], Brain_Data)

    # Logistic classificiation, with 2 fold cross-validation.
    stats = sim_brain_data.predict(
        algorithm="logistic", cv_dict={"type": "kfolds", "n_folds": 2}, plot=False
    )
    assert isinstance(stats["weight_map"], Brain_Data)

    # Ridge classificiation,
    stats = sim_brain_data.predict(
        algorithm="ridgeClassifier", cv_dict=None, plot=False
    )
    assert isinstance(stats["weight_map"], Brain_Data)

    # Ridge
    stats = sim_brain_data.predict(
        algorithm="ridge",
        cv_dict={"type": "kfolds", "n_folds": 2, "subject_id": holdout},
        plot=False,
        **{"alpha": 0.1}
    )

    # Lasso
    stats = sim_brain_data.predict(
        algorithm="lasso",
        cv_dict={"type": "kfolds", "n_folds": 2, "stratified": sim_brain_data.Y},
        plot=False,
        **{"alpha": 0.1}
    )

    # PCR
    stats = sim_brain_data.predict(algorithm="pcr", cv_dict=None, plot=False)


def test_predict_multi():
    # Simulate data 100 images worth
    sim = Simulator()
    sigma = 1
    y = [0, 1]
    n_reps = 50
    output_dir = "."
    dat = sim.create_data(y, sigma, reps=n_reps, output_dir=output_dir)
    y = pd.read_csv("y.csv", header=None, index_col=None)
    dat = Brain_Data("data.nii.gz", Y=y)

    # Predict within given ROIs
    # Generate some "rois" (in reality non-contiguous, but also not overlapping)
    roi_1 = dat[0].copy()
    roi_1.data = np.zeros_like(roi_1.data, dtype=bool)
    roi_2 = roi_1.copy()
    roi_3 = roi_1.copy()
    idx = np.random.choice(range(roi_1.shape()[-1]), size=9999, replace=False)
    roi_1.data[idx[:3333]] = 1
    roi_2.data[idx[3333:6666]] = 1
    roi_3.data[idx[6666:]] = 1
    rois = roi_1.append(roi_2).append(roi_3)

    # Load in all 50 rois so we can "insert" signal into the first one
    # rois = expand_mask(Brain_Data(os.path.join(get_resource_path(), 'k50.nii.gz')))
    # roi = rois[0]

    from sklearn.datasets import make_classification

    X, Y = make_classification(
        n_samples=100,
        n_features=rois[0].data.sum(),
        n_informative=500,
        n_redundant=5,
        n_classes=2,
    )
    dat.data[:, rois[0].data.astype(bool)] = X
    dat.Y = pd.Series(Y)

    out = dat.predict_multi(
        algorithm="svm",
        cv_dict={"type": "kfolds", "n_folds": 3},
        method="rois",
        n_jobs=-1,
        rois=rois[:3],
        kernel="linear",
    )
    assert len(out) == 3
    assert np.sum([elem["weight_map"].data.shape for elem in out]) == rois.data.sum()

    # Searchlight
    roi_mask = rois[:2].sum()
    out = dat.predict_multi(
        algorithm="svm",
        cv_dict={"type": "kfolds", "n_folds": 3},
        method="searchlight",
        radius=4,
        verbose=50,
        n_jobs=-1,
        process_mask=roi_mask,
    )
    assert len(np.nonzero(out.data)[0]) == len(np.nonzero(roi_mask.data)[0])


def test_similarity(sim_brain_data):
    stats = sim_brain_data.predict(
        algorithm="svm", cv_dict=None, plot=False, **{"kernel": "linear"}
    )
    r = sim_brain_data.similarity(stats["weight_map"])
    assert len(r) == shape_2d[0]
    r2 = sim_brain_data.similarity(stats["weight_map"].to_nifti())
    assert len(r2) == shape_2d[0]
    r = sim_brain_data.similarity(stats["weight_map"], method="dot_product")
    assert len(r) == shape_2d[0]
    r = sim_brain_data.similarity(stats["weight_map"], method="cosine")
    assert len(r) == shape_2d[0]
    r = sim_brain_data.similarity(sim_brain_data, method="correlation")
    assert r.shape == (sim_brain_data.shape()[0], sim_brain_data.shape()[0])
    r = sim_brain_data.similarity(sim_brain_data, method="dot_product")
    assert r.shape == (sim_brain_data.shape()[0], sim_brain_data.shape()[0])
    r = sim_brain_data.similarity(sim_brain_data, method="cosine")
    assert r.shape == (sim_brain_data.shape()[0], sim_brain_data.shape()[0])


def test_decompose(sim_brain_data):
    n_components = 3
    stats = sim_brain_data.decompose(
        algorithm="pca", axis="voxels", n_components=n_components
    )
    assert n_components == len(stats["components"])
    assert stats["weights"].shape == (len(sim_brain_data), n_components)

    stats = sim_brain_data.decompose(
        algorithm="ica", axis="voxels", n_components=n_components
    )
    assert n_components == len(stats["components"])
    assert stats["weights"].shape == (len(sim_brain_data), n_components)

    sim_brain_data.data = sim_brain_data.data + 2
    sim_brain_data.data[sim_brain_data.data < 0] = 0
    stats = sim_brain_data.decompose(
        algorithm="nnmf", axis="voxels", n_components=n_components
    )
    assert n_components == len(stats["components"])
    assert stats["weights"].shape == (len(sim_brain_data), n_components)

    stats = sim_brain_data.decompose(
        algorithm="fa", axis="voxels", n_components=n_components
    )
    assert n_components == len(stats["components"])
    assert stats["weights"].shape == (len(sim_brain_data), n_components)

    stats = sim_brain_data.decompose(
        algorithm="pca", axis="images", n_components=n_components
    )
    assert n_components == len(stats["components"])
    assert stats["weights"].shape == (len(sim_brain_data), n_components)

    stats = sim_brain_data.decompose(
        algorithm="ica", axis="images", n_components=n_components
    )
    assert n_components == len(stats["components"])
    assert stats["weights"].shape == (len(sim_brain_data), n_components)

    sim_brain_data.data = sim_brain_data.data + 2
    sim_brain_data.data[sim_brain_data.data < 0] = 0
    stats = sim_brain_data.decompose(
        algorithm="nnmf", axis="images", n_components=n_components
    )
    assert n_components == len(stats["components"])
    assert stats["weights"].shape == (len(sim_brain_data), n_components)

    stats = sim_brain_data.decompose(
        algorithm="fa", axis="images", n_components=n_components
    )
    assert n_components == len(stats["components"])
    assert stats["weights"].shape == (len(sim_brain_data), n_components)


def test_hyperalignment():
    sim = Simulator()
    y = [0, 1]
    n_reps = 10
    s1 = create_sphere([0, 0, 0], radius=3)
    d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
    d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
    d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)
    data = [d1, d2, d3]

    # Test procrustes using align
    out = align(data, method="procrustes")
    assert len(data) == len(out["transformed"])
    assert len(data) == len(out["transformation_matrix"])
    assert data[0].shape() == out["common_model"].shape()
    transformed = np.dot(d1.data, out["transformation_matrix"][0])
    centered = d1.data - np.mean(d1.data, 0)
    transformed = (
        np.dot(centered / np.linalg.norm(centered), out["transformation_matrix"][0])
        * out["scale"][0]
    )
    np.testing.assert_almost_equal(
        0, np.sum(out["transformed"][0].data - transformed), decimal=5
    )

    # Test deterministic brain_data
    bout = d1.align(out["common_model"], method="deterministic_srm")
    assert d1.shape() == bout["transformed"].shape()
    assert d1.shape() == bout["common_model"].shape()
    assert d1.shape()[1] == bout["transformation_matrix"].shape[0]
    btransformed = np.dot(d1.data, bout["transformation_matrix"])
    np.testing.assert_almost_equal(0, np.sum(bout["transformed"].data - btransformed))

    # Test deterministic brain_data
    bout = d1.align(out["common_model"], method="probabilistic_srm")
    assert d1.shape() == bout["transformed"].shape()
    assert d1.shape() == bout["common_model"].shape()
    assert d1.shape()[1] == bout["transformation_matrix"].shape[0]
    btransformed = np.dot(d1.data, bout["transformation_matrix"])
    np.testing.assert_almost_equal(0, np.sum(bout["transformed"].data - btransformed))

    # Test procrustes brain_data
    bout = d1.align(out["common_model"], method="procrustes")
    assert d1.shape() == bout["transformed"].shape()
    assert d1.shape() == bout["common_model"].shape()
    assert d1.shape()[1] == bout["transformation_matrix"].shape[0]
    centered = d1.data - np.mean(d1.data, 0)
    btransformed = (
        np.dot(centered / np.linalg.norm(centered), bout["transformation_matrix"])
        * bout["scale"]
    )
    np.testing.assert_almost_equal(
        0, np.sum(bout["transformed"].data - btransformed), decimal=5
    )
    np.testing.assert_almost_equal(
        0, np.sum(out["transformed"][0].data - bout["transformed"].data)
    )

    # Test over time
    sim = Simulator()
    y = [0, 1]
    n_reps = 10
    s1 = create_sphere([0, 0, 0], radius=5)
    d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
    d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
    d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)
    data = [d1, d2, d3]

    out = align(data, method="procrustes", axis=1)
    assert len(data) == len(out["transformed"])
    assert len(data) == len(out["transformation_matrix"])
    assert data[0].shape() == out["common_model"].shape()
    centered = data[0].data.T - np.mean(data[0].data.T, 0)
    transformed = (
        np.dot(centered / np.linalg.norm(centered), out["transformation_matrix"][0])
        * out["scale"][0]
    )
    np.testing.assert_almost_equal(
        0, np.sum(out["transformed"][0].data - transformed.T), decimal=5
    )

    bout = d1.align(out["common_model"], method="deterministic_srm", axis=1)
    assert d1.shape() == bout["transformed"].shape()
    assert d1.shape() == bout["common_model"].shape()
    assert d1.shape()[0] == bout["transformation_matrix"].shape[0]
    btransformed = np.dot(d1.data.T, bout["transformation_matrix"])
    np.testing.assert_almost_equal(0, np.sum(bout["transformed"].data - btransformed.T))

    bout = d1.align(out["common_model"], method="probabilistic_srm", axis=1)
    assert d1.shape() == bout["transformed"].shape()
    assert d1.shape() == bout["common_model"].shape()
    assert d1.shape()[0] == bout["transformation_matrix"].shape[0]
    btransformed = np.dot(d1.data.T, bout["transformation_matrix"])
    np.testing.assert_almost_equal(0, np.sum(bout["transformed"].data - btransformed.T))

    bout = d1.align(out["common_model"], method="procrustes", axis=1)
    assert d1.shape() == bout["transformed"].shape()
    assert d1.shape() == bout["common_model"].shape()
    assert d1.shape()[0] == bout["transformation_matrix"].shape[0]
    centered = d1.data.T - np.mean(d1.data.T, 0)
    btransformed = (
        np.dot(centered / np.linalg.norm(centered), bout["transformation_matrix"])
        * bout["scale"]
    )
    np.testing.assert_almost_equal(
        0, np.sum(bout["transformed"].data - btransformed.T), decimal=5
    )
    np.testing.assert_almost_equal(
        0, np.sum(out["transformed"][0].data - bout["transformed"].data)
    )

