import numpy as np
import pandas as pd
from nltools.stats import (one_sample_permutation,
                           two_sample_permutation,
                           correlation_permutation,
                           matrix_permutation,
                           jackknife_permutation,
                           downsample,
                           upsample,
                           winsorize,
                           align,
                           transform_pairwise,
                           _calc_pvalue,
                           find_spikes)
from nltools.simulator import Simulator
from nltools.mask import create_sphere
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform

# import pytest


def test_permutation():
    dat = np.random.multivariate_normal([2, 6], [[.5, 2], [.5, 3]], 1000)
    x = dat[:, 0]
    y = dat[:, 1]
    stats = two_sample_permutation(x, y, tail=1, n_permute=1000)
    assert (stats['mean'] < -2) & (stats['mean'] > -6) & (stats['p'] < .001)
    stats = one_sample_permutation(x-y, tail=1, n_permute=1000)
    assert (stats['mean'] < -2) & (stats['mean'] > -6) & (stats['p'] < .001)
    stats = correlation_permutation(x, y, metric='pearson', tail=1)
    assert (stats['correlation'] > .4) & (stats['correlation'] < .85) & (stats['p'] < .001)
    stats = correlation_permutation(x, y, metric='spearman', tail=1)
    assert (stats['correlation'] > .4) & (stats['correlation'] < .85) & (stats['p'] < .001)
    stats = correlation_permutation(x, y, metric='kendall', tail=2)
    assert (stats['correlation'] > .4) & (stats['correlation'] < .85) & (stats['p'] < .001)
    stats = correlation_permutation(x, y, metric='pearson', method='circle_shift', tail=1)
    assert (stats['correlation'] > .4) & (stats['correlation'] < .85) & (stats['p'] < .001)
    stats = correlation_permutation(x, y, metric='pearson', method='phase_randomize', tail=1)
    assert (stats['correlation'] > .4) & (stats['correlation'] < .85) & (stats['p'] < .001)
    # with pytest.raises(ValueError):
    # 	correlation_permutation(x, y, metric='kendall',tail=3)
    # with pytest.raises(ValueError):
    # 	correlation_permutation(x, y, metric='doesntwork',tail=3)
    s = np.random.normal(0, 1, 10000)
    two_sided = _calc_pvalue(all_p=s, stat=1.96, tail=2)
    upper_p = _calc_pvalue(all_p=s, stat=1.96, tail=1)
    lower_p = _calc_pvalue(all_p=s, stat=-1.96, tail=1)
    sum_p = upper_p + lower_p
    np.testing.assert_almost_equal(two_sided, sum_p, decimal=3)

    # Test matrix_permutation
    dat = np.random.multivariate_normal([2, 6], [[.5, 2], [.5, 3]], 190)
    x = squareform(dat[:, 0])
    y = squareform(dat[:, 1])
    stats = matrix_permutation(x, y, n_permute=1000)
    assert (stats['correlation'] > .4) & (stats['correlation'] < .85) & (stats['p'] < .001)

    # Test jackknife_permutation
    dat = np.random.multivariate_normal([5, 10, 15, 25, 35, 45],
                                        [[1, .2, .5, .7, .8, .9],
                                         [.2, 1, .4, .1, .1, .1],
                                         [.5, .4, 1, .1, .1, .1],
                                         [.7, .1, .1, 1, .3, .6],
                                         [.8, .1, .1, .3, 1, .5],
                                         [.9, .1, .1, .6, .5, 1]], 200)
    dat = dat + np.random.randn(dat.shape[0], dat.shape[1])*.5
    data1 = pairwise_distances(dat[0:100, :].T, metric='correlation')
    data2 = pairwise_distances(dat[100:, :].T, metric='correlation')

    stats = jackknife_permutation(data1, data2)
    print(stats)
    assert (stats['correlation'] >= .4) & (stats['correlation'] <= .99) & (stats['p'] <= .05)


def test_downsample():
    dat = pd.DataFrame()
    dat['x'] = range(0, 100)
    dat['y'] = np.repeat(range(1, 11), 10)
    assert((dat.groupby('y').mean().values.ravel() == downsample(data=dat['x'], sampling_freq=10, target=1, target_type='hz', method='mean').values).all)
    assert((dat.groupby('y').median().values.ravel() == downsample(data=dat['x'], sampling_freq=10, target=1, target_type='hz', method='median').values).all)
    # with pytest.raises(ValueError):
    # 	downsample(data=list(dat['x']),sampling_freq=10,target=1,target_type='hz',method='median')
    # with pytest.raises(ValueError):
    # 	downsample(data=dat['x'],sampling_freq=10,target=1,target_type='hz',method='doesnotwork')
    # with pytest.raises(ValueError):
    # 	downsample(data=dat['x'],sampling_freq=10,target=1,target_type='doesnotwork',method='median')


def test_upsample():
    dat = pd.DataFrame()
    dat['x'] = range(0, 100)
    dat['y'] = np.repeat(range(1, 11), 10)
    fs = 2
    us = upsample(dat, sampling_freq=1, target=fs, target_type='hz')
    assert(dat.shape[0]*fs-fs == us.shape[0])
    fs = 3
    us = upsample(dat, sampling_freq=1, target=fs, target_type='hz')
    assert(dat.shape[0]*fs-fs == us.shape[0])
    # with pytest.raises(ValueError):
    # 	upsample(dat,sampling_freq=1,target=fs,target_type='hz',method='doesnotwork')
    # with pytest.raises(ValueError):
    # 	upsample(dat,sampling_freq=1,target=fs,target_type='doesnotwork',method='linear')


def test_winsorize():
    outlier_test = pd.DataFrame([92, 19, 101, 58, 1053, 91, 26, 78, 10, 13,
                                 -40, 101, 86, 85, 15, 89, 89, 28, -5, 41])

    out = winsorize(outlier_test, cutoff={'quantile': [0.05, .95]},
                    replace_with_cutoff=False).values.squeeze()
    correct_result = np.array([92, 19, 101, 58, 101, 91, 26, 78, 10,
                               13, -5, 101, 86, 85, 15, 89, 89, 28,
                               -5, 41])
    assert(np.sum(out == correct_result) == 20)

    out = winsorize(outlier_test, cutoff={'std': [2, 2]},
                    replace_with_cutoff=False).values.squeeze()
    correct_result = np.array([92, 19, 101, 58, 101, 91, 26, 78, 10, 13,
                               -40, 101, 86, 85, 15, 89, 89, 28, -5, 41])
    assert(np.sum(out == correct_result) == 20)

    out = winsorize(outlier_test, cutoff={'std': [2, 2]},
                    replace_with_cutoff=True).values.squeeze()
    correct_result = np.array([92., 19., 101., 58., 556.97961997, 91., 26.,
                               78., 10., 13., -40., 101., 86., 85., 15., 89.,
                               89., 28., -5., 41.])
    assert(np.round(np.mean(out)) == np.round(np.mean(correct_result)))


def test_align():
    # Test hyperalignment matrix
    sim = Simulator()
    y = [0, 1]
    n_reps = 10
    s1 = create_sphere([0, 0, 0], radius=3)
    d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
    d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
    d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)

    data = [d1.data, d2.data, d3.data]
    out = align(data, method='deterministic_srm')
    assert len(data) == len(out['transformed'])
    assert len(data) == len(out['transformation_matrix'])
    assert data[0].shape == out['common_model'].shape
    transformed = np.dot(data[0], out['transformation_matrix'][0])
    np.testing.assert_almost_equal(np.sum(out['transformed'][0]-transformed.T), 0, decimal=5)
    assert len(out['isc']) == out['transformed'][0].shape[0]

    out = align(data, method='probabilistic_srm')
    assert len(data) == len(out['transformed'])
    assert len(data) == len(out['transformation_matrix'])
    assert data[0].shape == out['common_model'].shape
    transformed = np.dot(data[0], out['transformation_matrix'][0])
    np.testing.assert_almost_equal(np.sum(out['transformed'][0]-transformed.T), 0, decimal=5)
    assert len(out['isc']) == out['transformed'][0].shape[0]

    out2 = align(data, method='procrustes')
    assert len(data) == len(out2['transformed'])
    assert data[0].shape == out2['common_model'].shape
    assert len(data) == len(out2['transformation_matrix'])
    assert len(data) == len(out2['disparity'])
    centered = data[0]-np.mean(data[0], 0)
    transformed = (np.dot(centered/np.linalg.norm(centered), out2['transformation_matrix'][0])*out2['scale'][0])
    np.testing.assert_almost_equal(np.sum(out2['transformed'][0]-transformed.T), 0, decimal=5)
    assert out2['transformed'][0].shape == out2['transformed'][0].shape
    assert out2['transformation_matrix'][0].shape == out2['transformation_matrix'][0].shape
    assert len(out2['isc']) == out['transformed'][0].shape[0]

    # Test hyperalignment on Brain_Data
    data = [d1, d2, d3]
    out = align(data, method='deterministic_srm')
    assert len(data) == len(out['transformed'])
    assert len(data) == len(out['transformation_matrix'])
    assert data[0].shape() == out['common_model'].shape
    transformed = np.dot(d1.data, out['transformation_matrix'][0].data.T)
    np.testing.assert_almost_equal(np.sum(out['transformed'][0].data-transformed), 0, decimal=5)
    assert len(out['isc']) == out['transformed'][0].shape[1]

    out = align(data, method='probabilistic_srm')
    assert len(data) == len(out['transformed'])
    assert len(data) == len(out['transformation_matrix'])
    assert data[0].shape() == out['common_model'].shape
    transformed = np.dot(d1.data, out['transformation_matrix'][0].data.T)
    np.testing.assert_almost_equal(np.sum(out['transformed'][0].data-transformed), 0, decimal=5)
    assert len(out['isc']) == out['transformed'][0].shape[1]

    out2 = align(data, method='procrustes')
    assert len(data) == len(out2['transformed'])
    assert data[0].shape() == out2['common_model'].shape()
    assert len(data) == len(out2['transformation_matrix'])
    assert len(data) == len(out2['disparity'])
    centered = data[0].data-np.mean(data[0].data, 0)
    transformed = (np.dot(centered/np.linalg.norm(centered), out2['transformation_matrix'][0].data)*out2['scale'][0])
    np.testing.assert_almost_equal(np.sum(out2['transformed'][0].data-transformed), 0, decimal=5)
    assert out2['transformed'][0].shape() == out2['transformed'][0].shape()
    assert out2['transformation_matrix'][0].shape == out2['transformation_matrix'][0].shape
    assert len(out2['isc']) == out2['transformed'][0].shape()[1]

    # Test hyperalignment on matrix over time (axis=1)
    sim = Simulator()
    y = [0, 1]
    n_reps = 10
    s1 = create_sphere([0, 0, 0], radius=5)
    d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
    d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
    d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)
    data = [d1.data, d2.data, d3.data]

    out = align(data, method='deterministic_srm', axis=1)
    assert len(data) == len(out['transformed'])
    assert len(data) == len(out['transformation_matrix'])
    assert data[0].shape == out['common_model'].shape
    transformed = np.dot(data[0].T, out['transformation_matrix'][0].data)
    np.testing.assert_almost_equal(np.sum(out['transformed'][0]-transformed), 0, decimal=4)
    assert len(out['isc']) == out['transformed'][0].shape[1]

    out = align(data, method='probabilistic_srm', axis=1)
    assert len(data) == len(out['transformed'])
    assert len(data) == len(out['transformation_matrix'])
    assert data[0].shape == out['common_model'].shape
    transformed = np.dot(data[0].T, out['transformation_matrix'][0])
    np.testing.assert_almost_equal(np.sum(out['transformed'][0]-transformed), 0, decimal=4)
    assert len(out['isc']) == out['transformed'][0].shape[1]

    out2 = align(data, method='procrustes', axis=1)
    assert len(data) == len(out2['transformed'])
    assert data[0].shape == out2['common_model'].shape
    assert len(data) == len(out2['transformation_matrix'])
    assert len(data) == len(out2['disparity'])
    centered = data[0]-np.mean(data[0], 0)
    transformed = (np.dot((centered/np.linalg.norm(centered)).T, out2['transformation_matrix'][0].data)*out2['scale'][0])
    np.testing.assert_almost_equal(np.sum(out2['transformed'][0]-transformed), 0, decimal=4)
    assert out2['transformed'][0].shape == out2['transformed'][0].shape
    assert out2['transformation_matrix'][0].shape == out2['transformation_matrix'][0].shape
    assert len(out2['isc']) == out2['transformed'][0].shape[0]

    # Test hyperalignment on Brain_Data over time (axis=1)
    data = [d1, d2, d3]
    out = align(data, method='deterministic_srm', axis=1)
    assert len(data) == len(out['transformed'])
    assert len(data) == len(out['transformation_matrix'])
    assert data[0].shape() == out['common_model'].shape
    transformed = np.dot(d1.data.T, out['transformation_matrix'][0].data).T
    np.testing.assert_almost_equal(np.sum(out['transformed'][0].data-transformed), 0, decimal=5)
    assert len(out['isc']) == out['transformed'][0].shape[0]

    out = align(data, method='probabilistic_srm', axis=1)
    assert len(data) == len(out['transformed'])
    assert len(data) == len(out['transformation_matrix'])
    assert data[0].shape() == out['common_model'].shape
    transformed = np.dot(d1.data.T, out['transformation_matrix'][0].data).T
    np.testing.assert_almost_equal(np.sum(out['transformed'][0].data-transformed), 0, decimal=5)
    assert len(out['isc']) == out['transformed'][0].shape[0]

    out2 = align(data, method='procrustes', axis=1)
    assert len(data) == len(out2['transformed'])
    assert data[0].shape() == out2['common_model'].shape()
    assert len(data) == len(out2['transformation_matrix'])
    assert len(data) == len(out2['disparity'])
    centered = data[0].data.T-np.mean(data[0].data.T, 0)
    transformed = (np.dot(centered/np.linalg.norm(centered), out2['transformation_matrix'][0].data)*out2['scale'][0]).T
    np.testing.assert_almost_equal(np.sum(out2['transformed'][0].data-transformed), 0, decimal=5)
    assert out2['transformed'][0].shape() == out2['transformed'][0].shape()
    assert out2['transformation_matrix'][0].shape == out2['transformation_matrix'][0].shape
    assert len(out2['isc']) == out2['transformed'][0].shape()[1]


def test_transform_pairwise():
    n_features = 50
    n_samples = 100
    # Test without groups
    new_n_samples = int(n_samples * (n_samples-1) / 2)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples,)
    x_new, y_new = transform_pairwise(X, y)
    assert x_new.shape == (new_n_samples, n_features)
    assert y_new.shape == (new_n_samples,)
    assert y_new.ndim == 1
    # Test with groups
    n_subs = 4
    new_n_samples = int(n_subs * ((n_samples/n_subs)*(n_samples/n_subs-1))/2)
    groups = np.repeat(np.arange(1, 1+n_subs), n_samples/n_subs)
    y = np.vstack((y, groups)).T
    x_new, y_new = transform_pairwise(X, y)
    assert x_new.shape == (new_n_samples, n_features)
    assert y_new.shape == (new_n_samples, 2)
    assert y_new.ndim == 2
    a = y_new[:, 1] == np.repeat(np.arange(1, 1+n_subs), ((n_samples/n_subs)*(n_samples/n_subs-1))/2)
    assert a.all()

def test_find_spikes():
    sim = Simulator()
    y = [0, 1]
    n_reps = 50
    s1 = create_sphere([0, 0, 0], radius=3)
    d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)

    spikes = find_spikes(d1)
    assert isinstance(spikes, pd.DataFrame)
    assert spikes.shape[0] == len(d1)

    spikes = find_spikes(d1.to_nifti())
    assert isinstance(spikes, pd.DataFrame)
    assert spikes.shape[0] == len(d1)
