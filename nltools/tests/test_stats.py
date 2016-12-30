import numpy as np
import pandas as pd
from nltools.stats import one_sample_permutation,two_sample_permutation,correlation_permutation

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
