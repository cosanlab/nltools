import numpy as np
import pandas as pd
from nltools.stats import (one_sample_permutation,
							two_sample_permutation,
							correlation_permutation,
							downsample,
							upsample,
							winsorize,
							align)
from nltools.simulator import Simulator
from nltools.mask import create_sphere

def test_permutation():
	dat = np.random.multivariate_normal([2, 6], [[.5, 2], [.5, 3]], 1000)
	x = dat[:, 0]
	y = dat[:, 1]
	stats = two_sample_permutation(x, y)
	assert (stats['mean'] < -2) & (stats['mean'] > -6)
	assert stats['p'] < .001
	print(stats)
	stats = one_sample_permutation(x-y)
	assert (stats['mean'] < -2) & (stats['mean'] > -6)
	assert stats['p'] < .001
	print(stats)
	stats = correlation_permutation(x, y, metric='pearson')
	assert (stats['correlation'] > .4) & (stats['correlation']<.85)
	assert stats['p'] < .001
	stats = correlation_permutation(x, y, metric='spearman')
	assert (stats['correlation'] > .4) & (stats['correlation']<.85)
	assert stats['p'] < .001
	stats = correlation_permutation(x, y, metric='kendall')
	assert (stats['correlation'] > .4) & (stats['correlation']<.85)
	assert stats['p'] < .001

def test_downsample():
	dat = pd.DataFrame()
	dat['x'] = range(0,100)
	dat['y'] = np.repeat(range(1,11),10)
	assert((dat.groupby('y').mean().values.ravel() == downsample(data=dat['x'],sampling_freq=10,target=1,target_type='hz',method='mean').values).all)
	assert((dat.groupby('y').median().values.ravel() == downsample(data=dat['x'],sampling_freq=10,target=1,target_type='hz',method='median').values).all)

def test_upsample():
	dat = pd.DataFrame()
	dat['x'] = range(0,100)
	dat['y'] = np.repeat(range(1,11),10)
	fs = 2
	us = upsample(dat,sampling_freq=1,target=fs,target_type='hz')
	assert(dat.shape[0]*fs-fs == us.shape[0])
	fs = 3
	us = upsample(dat,sampling_freq=1,target=fs,target_type='hz')
	assert(dat.shape[0]*fs-fs == us.shape[0])

def test_winsorize():
	outlier_test = pd.DataFrame([92, 19, 101, 58, 1053, 91, 26, 78, 10, 13,
								-40, 101, 86, 85, 15, 89, 89, 28, -5, 41])

	out = winsorize(outlier_test,cutoff={'quantile':[0.05, .95]},
					replace_with_cutoff=False).values.squeeze()
	correct_result = np.array([92, 19, 101, 58, 101, 91, 26, 78, 10,
								13, -5, 101, 86, 85, 15, 89, 89, 28,
								-5, 41])
	assert(np.sum(out == correct_result) == 20)

	out = winsorize(outlier_test,cutoff={'std':[2, 2]},
					replace_with_cutoff=False).values.squeeze()
	correct_result = np.array([92, 19, 101, 58, 101, 91, 26, 78, 10, 13,
								-40, 101, 86, 85, 15, 89, 89, 28, -5, 41])
	assert(np.sum(out==correct_result)==20)

	out = winsorize(outlier_test,cutoff={'std':[2, 2]},
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

	data = [d1.data.T,d2.data.T,d3.data.T]
	out = align(data, method='deterministic_srm')
	assert len(data) == len(out['transformed'])
	assert len(data) == len(out['transformation_matrix'])
	assert data[0].shape == out['common_model'].shape
	transformed = np.dot(data[0].T,out['transformation_matrix'][0])
	np.testing.assert_almost_equal(0,np.sum(out['transformed'][0]-transformed.T))

	out = align(data, method='probabilistic_srm')
	assert len(data) == len(out['transformed'])
	assert len(data) == len(out['transformation_matrix'])
	assert data[0].shape == out['common_model'].shape
	transformed = np.dot(data[0].T,out['transformation_matrix'][0])
	np.testing.assert_almost_equal(0,np.sum(out['transformed'][0]-transformed.T))

	out2 = align(data, method='procrustes')
	assert len(data) == len(out2['transformed'])
	assert data[0].shape == out2['common_model'].shape
	assert len(data) == len(out2['transformation_matrix'])
	assert len(data) == len(out2['disparity'])
	centered = data[0].T-np.mean(data[0].T,0)
	transformed = (np.dot(centered/np.linalg.norm(centered), out2['transformation_matrix'][0])*out2['scale'][0])
	np.testing.assert_almost_equal(0,np.sum(out2['transformed'][0]-transformed.T))
	assert out['transformed'][0].shape == out2['transformed'][0].shape
	assert out['transformation_matrix'][0].shape == out2['transformation_matrix'][0].shape

	# Test hyperalignment on Brain_Data
	data = [d1,d2,d3]
	out = align(data, method='deterministic_srm')
	assert len(data) == len(out['transformed'])
	assert len(data) == len(out['transformation_matrix'])
	assert data[0].shape() == out['common_model'].shape()
	transformed = np.dot(d1.data,out['transformation_matrix'][0])
	np.testing.assert_almost_equal(0,np.sum(out['transformed'][0].data-transformed))

	out = align(data, method='probabilistic_srm')
	assert len(data) == len(out['transformed'])
	assert len(data) == len(out['transformation_matrix'])
	assert data[0].shape() == out['common_model'].shape()
	transformed = np.dot(d1.data,out['transformation_matrix'][0])
	np.testing.assert_almost_equal(0,np.sum(out['transformed'][0].data-transformed))

	out2 = align(data, method='procrustes')
	assert len(data) == len(out2['transformed'])
	assert data[0].shape() == out2['common_model'].shape()
	assert len(data) == len(out2['transformation_matrix'])
	assert len(data) == len(out2['disparity'])
	centered = data[0].data-np.mean(data[0].data,0)
	transformed = (np.dot(centered/np.linalg.norm(centered), out2['transformation_matrix'][0])*out2['scale'][0])
	np.testing.assert_almost_equal(0,np.sum(out2['transformed'][0].data-transformed))
	assert out['transformed'][0].shape() == out2['transformed'][0].shape()
	assert out['transformation_matrix'][0].shape == out2['transformation_matrix'][0].shape

	# Test hyperalignment on matrix over time (axis=1)
	sim = Simulator()
	y = [0, 1]
	n_reps = 10
	s1 = create_sphere([0, 0, 0], radius=5)
	d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
	d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
	d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)
	data = [d1.data.T,d2.data.T,d3.data.T]
	out = align(data, method='deterministic_srm', axis=1)
	assert len(data) == len(out['transformed'])
	assert len(data) == len(out['transformation_matrix'])
	assert data[0].shape == out['common_model'].shape
	transformed = np.dot(data[0],out['transformation_matrix'][0])
	np.testing.assert_almost_equal(0,np.sum(out['transformed'][0]-transformed))

	out = align(data, method='probabilistic_srm', axis=1)
	assert len(data) == len(out['transformed'])
	assert len(data) == len(out['transformation_matrix'])
	assert data[0].shape == out['common_model'].shape
	transformed = np.dot(data[0],out['transformation_matrix'][0])
	np.testing.assert_almost_equal(0,np.sum(out['transformed'][0]-transformed))

	out2 = align(data, method='procrustes', axis=1)
	assert len(data) == len(out2['transformed'])
	assert data[0].shape == out2['common_model'].shape
	assert len(data) == len(out2['transformation_matrix'])
	assert len(data) == len(out2['disparity'])
	centered = data[0]-np.mean(data[0],0)
	transformed = (np.dot(centered/np.linalg.norm(centered), out2['transformation_matrix'][0])*out2['scale'][0])
	np.testing.assert_almost_equal(0,np.sum(out2['transformed'][0]-transformed))
	assert out['transformed'][0].shape == out2['transformed'][0].shape
	assert out['transformation_matrix'][0].shape == out2['transformation_matrix'][0].shape

	# Test hyperalignment on Brain_Data over time (axis=1)
	data = [d1, d2, d3]
	out = align(data, method='deterministic_srm', axis=1)
	assert len(data) == len(out['transformed'])
	assert len(data) == len(out['transformation_matrix'])
	assert data[0].shape() == out['common_model'].shape()
	transformed = np.dot(d1.data.T,out['transformation_matrix'][0])
	np.testing.assert_almost_equal(0,np.sum(out['transformed'][0].data-transformed.T))

	out = align(data, method='probabilistic_srm', axis=1)
	assert len(data) == len(out['transformed'])
	assert len(data) == len(out['transformation_matrix'])
	assert data[0].shape() == out['common_model'].shape()
	transformed = np.dot(d1.data.T,out['transformation_matrix'][0])
	np.testing.assert_almost_equal(0,np.sum(out['transformed'][0].data-transformed.T))

	out2 = align(data, method='procrustes', axis=1)
	assert len(data) == len(out2['transformed'])
	assert data[0].shape() == out2['common_model'].shape()
	assert len(data) == len(out2['transformation_matrix'])
	assert len(data) == len(out2['disparity'])
	centered = data[0].data.T-np.mean(data[0].data.T,0)
	transformed = (np.dot(centered/np.linalg.norm(centered), out2['transformation_matrix'][0])*out2['scale'][0])
	np.testing.assert_almost_equal(0,np.sum(out2['transformed'][0].data-transformed.T))
	assert out['transformed'][0].shape() == out2['transformed'][0].shape()
	assert out['transformation_matrix'][0].shape == out2['transformation_matrix'][0].shape
	
