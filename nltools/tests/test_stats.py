import numpy as np
import pandas as pd
from nltools.stats import one_sample_permutation,two_sample_permutation,correlation_permutation, downsample

def test_permutation():
	dat = np.random.multivariate_normal([2,6],[[.5,2],[.5,3]],100)
	x = dat[:,0]
	y = dat[:,1]
	stats = two_sample_permutation(x,y)
	assert (stats['mean'] < -2) & (stats['mean'] > -6)
	assert stats['p']< .001
	print(stats)
	stats = one_sample_permutation(x-y)
	assert (stats['mean'] < -2) & (stats['mean'] > -6)
	assert stats['p']< .001
	print(stats)
	stats = correlation_permutation(x,y)
	assert (stats['correlation']>.4) & (stats['correlation']<.85)
	assert stats['p']< .001
	stats = correlation_permutation(x,y,metric='kendall')
	assert (stats['correlation']>.4) & (stats['correlation']<.85)
	assert stats['p']< .001

def test_downsample():
	dat = pd.DataFrame()
	dat['x'] = range(0,100)
	dat['y'] = np.repeat(range(1,11),10)
	fs = 2
	us = upsample(dat,sampling_freq=1,target=fs,target_type='hz')
	assert(dat.shape[0]*fs-fs,us.shape[0])
	fs = 3
	us = upsample(dat,sampling_freq=1,target=fs,target_type='hz')
	assert(dat.shape[0]*fs-fs,us.shape[0])

def test_upsample():
