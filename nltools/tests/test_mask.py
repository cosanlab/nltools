from nltools.mask import create_sphere

def test_create_sphere():
	a = create_sphere(radius=10, coordinates=[0, 0, 0])
	assert a.sum() >= 515
	a = create_sphere(radius=[10,5], coordinates=[[0, 0, 0],
						[15, 0, 25]])
	assert a.sum() >= 571
	a = create_sphere(radius=10, coordinates=[[0, 0, 0],
						[15, 0, 25]])
	assert a.sum() >= 1051
