from nltools.mask import create_sphere
import numpy as np

def test_create_sphere():
	# Test values update to reflect the fact that standard Brain_Data mask has few voxels because ventricles are 0'd out
	a = create_sphere(radius=10, coordinates=[0, 0, 0])
	assert np.sum(a.get_data()) >= 497 #515
	a = create_sphere(radius=[10, 5], coordinates=[[0, 0, 0],
	                    [15, 0, 25]])
	assert np.sum(a.get_data()) >= 553 #571
	a = create_sphere(radius=10, coordinates=[[0, 0, 0],
	                    [15, 0, 25]])
	assert np.sum(a.get_data()) >= 1013 #1051
