import numpy as np
import pandas as pd
from nltools.stats import (one_sample_permutation,
							two_sample_permutation,
							correlation_permutation,
							downsample,
							upsample,
							winsorize,
							align)

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
	out = align(data, method='deterministic_srm')
	print(out.keys())
	assert len(data) == len(out['transformed_data'])
	assert len(data) == len(out['transformation_matrices'])
	assert data[0].shape == out['common_model'].shape

	out = align(data, method='probabilistic_srm')
	print(out.keys())
	assert len(data) == len(out['transformed_data'])
	assert len(data) == len(out['transformation_matrices'])
	assert data[0].shape == out['common_model'].shape

	out = align(data, method='hyperalignment')
	print(out.keys())
	assert len(data) == len(out['transformed_data'])
	assert data[0].shape == out['common_model'].shape
