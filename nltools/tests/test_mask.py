from nltools.mask import create_sphere
import numpy as np

def test_create_sphere():
	a = create_sphere(radius=10, coordinates=[0, 0, 0])
	assert np.sum(a.get_data()) >= 515
	a = create_sphere(radius=[10, 5], coordinates=[[0, 0, 0],
	                    [15, 0, 25]])
	assert np.sum(a.get_data()) >= 571
	a = create_sphere(radius=10, coordinates=[[0, 0, 0],
	                    [15, 0, 25]])
	assert np.sum(a.get_data()) >= 1051
