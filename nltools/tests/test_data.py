import os
import numpy as np
import nibabel as nb
import pandas as pd
import glob
from nltools.simulator import Simulator
from nltools.data import (Brain_Data,
                        Adjacency,
                        Groupby,
                        Design_Matrix)
from nltools.stats import threshold, align
from nltools.mask import create_sphere
from nltools.external.hrf import glover_hrf
from sklearn.metrics import pairwise_distances
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import six
from nltools.prefs import MNI_Template
from scipy.stats import pearsonr
from scipy.linalg import block_diag

matplotlib.use('TkAgg')

def test_brain_data(tmpdir):

    # Add 3mm to list to test that resolution as well
    for resolution in ['2mm']:

        MNI_Template["resolution"] = resolution

        sim = Simulator()
        r = 10
        sigma = 1
        y = [0, 1]
        n_reps = 3
        output_dir = str(tmpdir)
        dat = sim.create_data(y, sigma, reps=n_reps, output_dir=output_dir)

        if MNI_Template["resolution"] == '2mm':
            shape_3d = (91, 109, 91)
            shape_2d = (6, 238955)
        elif MNI_Template["resolution"] == '3mm':
            shape_3d = (60, 72, 60)
            shape_2d = (6, 71020)

        y = pd.read_csv(os.path.join(str(tmpdir.join('y.csv'))),header=None, index_col=None)
        holdout = pd.read_csv(os.path.join(str(tmpdir.join('rep_id.csv'))),header=None,index_col=None)

        # Test load list of 4D images
        file_list = [str(tmpdir.join('data.nii.gz')), str(tmpdir.join('data.nii.gz'))]
        dat = Brain_Data(file_list)
        dat = Brain_Data([nb.load(x) for x in file_list])

        # Test load list
        dat = Brain_Data(data=str(tmpdir.join('data.nii.gz')), Y=y)

        # Test concatenate
        out = Brain_Data([x for x in dat])
        assert isinstance(out, Brain_Data)
        assert len(out)==len(dat)

        # Test to_nifti
        d = dat.to_nifti()
        assert d.shape[0:3] == shape_3d

        # Test load nibabel
        assert Brain_Data(d)

        # Test shape
        assert dat.shape() == shape_2d

        # Test Mean
        assert dat.mean().shape()[0] == shape_2d[1]

        # Test Std
        assert dat.std().shape()[0] == shape_2d[1]

        # Test add
        new = dat + dat
        assert new.shape() == shape_2d

        # Test subtract
        new = dat - dat
        assert new.shape() == shape_2d

        # Test multiply
        new = dat * dat
        assert new.shape() == shape_2d

        # Test Indexing
        index = [0, 3, 1]
        assert len(dat[index]) == len(index)
        index = range(4)
        assert len(dat[index]) == len(index)
        index = dat.Y == 1

        assert len(dat[index.values.flatten()]) == index.values.sum()

        assert len(dat[index]) == index.values.sum()
        assert len(dat[:3]) == 3

        # Test Iterator
        x = [x for x in dat]
        assert len(x) == len(dat)
        assert len(x[0].data.shape) == 1

        # # Test T-test
        out = dat.ttest()
        assert out['t'].shape()[0] == shape_2d[1]

        # # # Test T-test - permutation method
        # out = dat.ttest(threshold_dict={'permutation':'tfce','n_permutations':50,'n_jobs':1})
        # assert out['t'].shape()[0]==shape_2d[1]

        # Test Regress
        dat.X = pd.DataFrame({'Intercept':np.ones(len(dat.Y)),
                            'X1':np.array(dat.Y).flatten()}, index=None)

        # Standard OLS
        out = dat.regress()

        assert type(out['beta'].data) == np.ndarray
        assert type(out['t'].data) == np.ndarray
        assert type(out['p'].data) == np.ndarray
        assert type(out['residual'].data) == np.ndarray
        assert type(out['df'].data) == np.ndarray
        assert out['beta'].shape() == (2, shape_2d[1])
        assert out['t'][1].shape()[0] == shape_2d[1]

        # Robust OLS
        out = dat.regress(mode='robust')

        assert type(out['beta'].data) == np.ndarray
        assert type(out['t'].data) == np.ndarray
        assert type(out['p'].data) == np.ndarray
        assert type(out['residual'].data) == np.ndarray
        assert type(out['df'].data) == np.ndarray
        assert out['beta'].shape() == (2, shape_2d[1])
        assert out['t'][1].shape()[0] == shape_2d[1]

        # Test threshold
        i=1
        tt = threshold(out['t'][i], out['p'][i], .05)
        assert isinstance(tt, Brain_Data)

        # Test write
        dat.write(os.path.join(str(tmpdir.join('test_write.nii'))))
        assert Brain_Data(os.path.join(str(tmpdir.join('test_write.nii'))))

        # Test append
        assert dat.append(dat).shape()[0] == shape_2d[0]*2

        # Test distance
        distance = dat.distance(method='euclidean')
        assert isinstance(distance, Adjacency)
        assert distance.square_shape()[0] == shape_2d[0]

        # Test predict
        stats = dat.predict(algorithm='svm',
                            cv_dict={'type': 'kfolds', 'n_folds': 2},
                            plot=False, **{'kernel':"linear"})

        # Support Vector Regression, with 5 fold cross-validation with Platt Scaling
        # This will output probabilities of each class
        stats = dat.predict(algorithm='svm',
                            cv_dict=None, plot=False,
                            **{'kernel':'linear', 'probability':True})
        assert isinstance(stats['weight_map'], Brain_Data)

        # Logistic classificiation, with 2 fold cross-validation.
        stats = dat.predict(algorithm='logistic',
                            cv_dict={'type': 'kfolds', 'n_folds': 2},
                            plot=False)
        assert isinstance(stats['weight_map'], Brain_Data)

        # Ridge classificiation,
        stats = dat.predict(algorithm='ridgeClassifier', cv_dict=None, plot=False)
        assert isinstance(stats['weight_map'], Brain_Data)

        # Ridge
        stats = dat.predict(algorithm='ridge',
                            cv_dict={'type': 'kfolds', 'n_folds': 2,
                            'subject_id':holdout}, plot=False, **{'alpha':.1})

        # Lasso
        stats = dat.predict(algorithm='lasso',
                            cv_dict={'type': 'kfolds', 'n_folds': 2,
                            'stratified':dat.Y}, plot=False, **{'alpha':.1})

        # PCR
        stats = dat.predict(algorithm='pcr', cv_dict=None, plot=False)

        # Test Similarity
        r = dat.similarity(stats['weight_map'])
        assert len(r) == shape_2d[0]
        r2 = dat.similarity(stats['weight_map'].to_nifti())
        assert len(r2) == shape_2d[0]
        r = dat.similarity(stats['weight_map'], method='dot_product')
        assert len(r) == shape_2d[0]
        r = dat.similarity(stats['weight_map'], method='cosine')
        assert len(r) == shape_2d[0]
        r = dat.similarity(dat, method='correlation')
        assert r.shape == (dat.shape()[0],dat.shape()[0])
        r = dat.similarity(dat, method='dot_product')
        assert r.shape == (dat.shape()[0],dat.shape()[0])
        r = dat.similarity(dat, method='cosine')
        assert r.shape == (dat.shape()[0],dat.shape()[0])

        # Test apply_mask - might move part of this to test mask suite
        s1 = create_sphere([12, 10, -8], radius=10)
        assert isinstance(s1, nb.Nifti1Image)
        masked_dat = dat.apply_mask(s1)
        assert masked_dat.shape()[1] == np.sum(s1.get_data() != 0)

        # Test extract_roi
        mask = create_sphere([12, 10, -8], radius=10)
        assert len(dat.extract_roi(mask)) == shape_2d[0]

        # Test r_to_z
        z = dat.r_to_z()
        assert z.shape() == dat.shape()

        # Test copy
        d_copy = dat.copy()
        assert d_copy.shape() == dat.shape()

        # Test detrend
        detrend = dat.detrend()
        assert detrend.shape() == dat.shape()

        # Test standardize
        s = dat.standardize()
        assert s.shape() == dat.shape()
        assert np.isclose(np.sum(s.mean().data), 0, atol=.1)
        s = dat.standardize(method='zscore')
        assert s.shape() == dat.shape()
        assert np.isclose(np.sum(s.mean().data), 0, atol=.1)

        # Test Sum
        s = dat.sum()
        assert s.shape() == dat[1].shape()

        # Test Groupby
        s1 = create_sphere([12, 10, -8], radius=10)
        s2 = create_sphere([22, -2, -22], radius=10)
        mask = Brain_Data([s1, s2])
        d = dat.groupby(mask)
        assert isinstance(d, Groupby)

        # Test Aggregate
        mn = dat.aggregate(mask, 'mean')
        assert isinstance(mn, Brain_Data)
        assert len(mn.shape()) == 1

        # Test Threshold
        s1 = create_sphere([12, 10, -8], radius=10)
        s2 = create_sphere([22, -2, -22], radius=10)
        mask = Brain_Data(s1)*5
        mask = mask + Brain_Data(s2)

        m1 = mask.threshold(upper=.5)
        m2 = mask.threshold(upper=3)
        m3 = mask.threshold(upper='98%')
        m4 = Brain_Data(s1)*5 + Brain_Data(s2)*-.5
        m4 = mask.threshold(upper=.5,lower=-.3)
        assert np.sum(m1.data > 0) > np.sum(m2.data > 0)
        assert np.sum(m1.data > 0) == np.sum(m3.data > 0)
        assert np.sum(m4.data[(m4.data > -.3) & (m4.data <.5)]) == 0
        assert np.sum(m4.data[(m4.data < -.3) | (m4.data >.5)]) > 0

        # Test Regions
        r = mask.regions(min_region_size=10)
        m1 = Brain_Data(s1)
        m2 = r.threshold(1, binarize=True)
        # assert len(r)==2
        assert len(np.unique(r.to_nifti().get_data())) == 2
        diff = m2-m1
        assert np.sum(diff.data) == 0

        # Test Bootstrap
        masked = dat.apply_mask(create_sphere(radius=10, coordinates=[0, 0, 0]))
        n_samples = 3
        b = masked.bootstrap('mean', n_samples=n_samples)
        assert isinstance(b['Z'], Brain_Data)
        b = masked.bootstrap('std', n_samples=n_samples)
        assert isinstance(b['Z'], Brain_Data)
        b = masked.bootstrap('predict', n_samples=n_samples, plot=False)
        assert isinstance(b['Z'], Brain_Data)
        b = masked.bootstrap('predict', n_samples=n_samples,
                        plot=False, cv_dict={'type':'kfolds','n_folds':3})
        assert isinstance(b['Z'], Brain_Data)
        b = masked.bootstrap('predict', n_samples=n_samples,
                        save_weights=True, plot=False)
        assert len(b['samples'])==n_samples

        # Test decompose
        n_components = 3
        stats = dat.decompose(algorithm='pca', axis='voxels',
                              n_components=n_components)
        assert n_components == len(stats['components'])
        assert stats['weights'].shape == (len(dat), n_components)

        stats = dat.decompose(algorithm='ica', axis='voxels',
                              n_components=n_components)
        assert n_components == len(stats['components'])
        assert stats['weights'].shape == (len(dat), n_components)

        dat.data = dat.data + 2
        dat.data[dat.data<0] = 0
        stats = dat.decompose(algorithm='nnmf', axis='voxels',
                              n_components=n_components)
        assert n_components == len(stats['components'])
        assert stats['weights'].shape == (len(dat), n_components)

        stats = dat.decompose(algorithm='fa', axis='voxels',
                              n_components=n_components)
        assert n_components == len(stats['components'])
        assert stats['weights'].shape == (len(dat), n_components)

        stats = dat.decompose(algorithm='pca', axis='images',
                              n_components=n_components)
        assert n_components == len(stats['components'])
        assert stats['weights'].shape == (len(dat), n_components)

        stats = dat.decompose(algorithm='ica', axis='images',
                              n_components=n_components)
        assert n_components == len(stats['components'])
        assert stats['weights'].shape == (len(dat), n_components)

        dat.data = dat.data + 2
        dat.data[dat.data<0] = 0
        stats = dat.decompose(algorithm='nnmf', axis='images',
                              n_components=n_components)
        assert n_components == len(stats['components'])
        assert stats['weights'].shape == (len(dat), n_components)

        stats = dat.decompose(algorithm='fa', axis='images',
                              n_components=n_components)
        assert n_components == len(stats['components'])
        assert stats['weights'].shape == (len(dat), n_components)

        # Test Hyperalignment Method
        sim = Simulator()
        y = [0, 1]
        n_reps = 10
        s1 = create_sphere([0, 0, 0], radius=3)
        d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
        d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
        d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)

        # Test procrustes using align
        data = [d1, d2, d3]
        out = align(data, method='procrustes')
        assert len(data) == len(out['transformed'])
        assert len(data) == len(out['transformation_matrix'])
        assert data[0].shape() == out['common_model'].shape()
        transformed = np.dot(d1.data, out['transformation_matrix'][0])
        centered = d1.data - np.mean(d1.data, 0)
        transformed = (np.dot(centered/np.linalg.norm(centered), out['transformation_matrix'][0])*out['scale'][0])
        np.testing.assert_almost_equal(0, np.sum(out['transformed'][0].data - transformed), decimal=5)

        # Test deterministic brain_data
        bout = d1.align(out['common_model'], method='deterministic_srm')
        assert d1.shape() == bout['transformed'].shape()
        assert d1.shape() == bout['common_model'].shape()
        assert d1.shape()[1] == bout['transformation_matrix'].shape[0]
        btransformed = np.dot(d1.data, bout['transformation_matrix'])
        np.testing.assert_almost_equal(0, np.sum(bout['transformed'].data - btransformed))

        # Test deterministic brain_data
        bout = d1.align(out['common_model'], method='probabilistic_srm')
        assert d1.shape() == bout['transformed'].shape()
        assert d1.shape() == bout['common_model'].shape()
        assert d1.shape()[1] == bout['transformation_matrix'].shape[0]
        btransformed = np.dot(d1.data, bout['transformation_matrix'])
        np.testing.assert_almost_equal(0, np.sum(bout['transformed'].data-btransformed))

        # Test procrustes brain_data
        bout = d1.align(out['common_model'], method='procrustes')
        assert d1.shape() == bout['transformed'].shape()
        assert d1.shape() == bout['common_model'].shape()
        assert d1.shape()[1] == bout['transformation_matrix'].shape[0]
        centered = d1.data - np.mean(d1.data, 0)
        btransformed = (np.dot(centered/np.linalg.norm(centered), bout['transformation_matrix'])*bout['scale'])
        np.testing.assert_almost_equal(0, np.sum(bout['transformed'].data-btransformed), decimal=5)
        np.testing.assert_almost_equal(0, np.sum(out['transformed'][0].data - bout['transformed'].data))

        # Test hyperalignment on Brain_Data over time (axis=1)
        sim = Simulator()
        y = [0, 1]
        n_reps = 10
        s1 = create_sphere([0, 0, 0], radius=5)
        d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
        d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
        d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)
        data = [d1, d2, d3]

        out = align(data, method='procrustes', axis=1)
        assert len(data) == len(out['transformed'])
        assert len(data) == len(out['transformation_matrix'])
        assert data[0].shape() == out['common_model'].shape()
        centered = data[0].data.T-np.mean(data[0].data.T, 0)
        transformed = (np.dot(centered/np.linalg.norm(centered), out['transformation_matrix'][0])*out['scale'][0])
        np.testing.assert_almost_equal(0,np.sum(out['transformed'][0].data-transformed.T), decimal=5)

        bout = d1.align(out['common_model'], method='deterministic_srm', axis=1)
        assert d1.shape() == bout['transformed'].shape()
        assert d1.shape() == bout['common_model'].shape()
        assert d1.shape()[0] == bout['transformation_matrix'].shape[0]
        btransformed = np.dot(d1.data.T, bout['transformation_matrix'])
        np.testing.assert_almost_equal(0, np.sum(bout['transformed'].data-btransformed.T))

        bout = d1.align(out['common_model'], method='probabilistic_srm', axis=1)
        assert d1.shape() == bout['transformed'].shape()
        assert d1.shape() == bout['common_model'].shape()
        assert d1.shape()[0] == bout['transformation_matrix'].shape[0]
        btransformed = np.dot(d1.data.T, bout['transformation_matrix'])
        np.testing.assert_almost_equal(0, np.sum(bout['transformed'].data-btransformed.T))

        bout = d1.align(out['common_model'], method='procrustes', axis=1)
        assert d1.shape() == bout['transformed'].shape()
        assert d1.shape() == bout['common_model'].shape()
        assert d1.shape()[0] == bout['transformation_matrix'].shape[0]
        centered = d1.data.T-np.mean(d1.data.T, 0)
        btransformed = (np.dot(centered/np.linalg.norm(centered), bout['transformation_matrix'])*bout['scale'])
        np.testing.assert_almost_equal(0, np.sum(bout['transformed'].data-btransformed.T), decimal=5)
        np.testing.assert_almost_equal(0, np.sum(out['transformed'][0].data-bout['transformed'].data))


def test_adjacency(tmpdir):
    n = 10
    sim = np.random.multivariate_normal([0,0,0,0],[[1, 0.8, 0.1, 0.4],
                                         [0.8, 1, 0.6, 0.1],
                                         [0.1, 0.6, 1, 0.3],
                                         [0.4, 0.1, 0.3, 1]], 100)
    data = pairwise_distances(sim.T, metric='correlation')
    dat_all = []
    for t in range(n):
        tmp = data
        dat_all.append(tmp)
    sim_directed = np.array([[1, 0.5, 0.3, 0.4],
              [0.8, 1, 0.2, 0.1],
              [0.7, 0.6, 1, 0.5],
              [0.85, 0.4, 0.3, 1]])
    labels = ['v_%s' % (x+1) for x in range(sim.shape[1])]
    dat_single = Adjacency(dat_all[0], labels=labels)
    dat_multiple = Adjacency(dat_all, labels=labels)
    dat_directed = Adjacency(sim_directed, matrix_type='directed',
                             labels=labels)

    # Test automatic distance/similarity detection
    assert dat_single.matrix_type is 'distance'
    dat_single2 = Adjacency(1-data)
    assert dat_single2.matrix_type is 'similarity'
    assert not dat_directed.issymmetric
    assert dat_single.issymmetric

    # Test length
    assert len(dat_multiple) == dat_multiple.data.shape[0]
    assert len(dat_multiple[0]) == 1

    # Test Indexing
    assert len(dat_multiple[0]) == 1
    assert len(dat_multiple[0:4]) == 4
    assert len(dat_multiple[0, 2, 3]) == 3

    # Test basic arithmetic
    assert(dat_directed+5).data[0] == dat_directed.data[0]+5
    assert(dat_directed-.5).data[0] == dat_directed.data[0]-.5
    assert(dat_directed*5).data[0] == dat_directed.data[0]*5
    assert np.all(np.isclose((dat_directed+dat_directed).data,
                (dat_directed*2).data))
    assert np.all(np.isclose((dat_directed*2-dat_directed).data,
                dat_directed.data))

    # Test copy
    assert np.all(dat_multiple.data == dat_multiple.copy().data)

    # Test squareform & iterable
    assert len(dat_multiple.squareform()) == len(dat_multiple)
    assert dat_single.squareform().shape == data.shape
    assert dat_directed.squareform().shape == sim_directed.shape

    # Test write
    dat_multiple.write(os.path.join(str(tmpdir.join('Test.csv'))),
                        method='long')
    dat_multiple2 = Adjacency(os.path.join(str(tmpdir.join('Test.csv'))),
                        matrix_type='distance_flat')
    dat_directed.write(os.path.join(str(tmpdir.join('Test.csv'))),
                        method='long')
    dat_directed2 = Adjacency(os.path.join(str(tmpdir.join('Test.csv'))),
                        matrix_type='directed_flat')
    assert np.all(np.isclose(dat_multiple.data, dat_multiple2.data))
    assert np.all(np.isclose(dat_directed.data, dat_directed2.data))

    # Test mean
    assert isinstance(dat_multiple.mean(axis=0), Adjacency)
    assert len(dat_multiple.mean(axis=0)) == 1
    assert len(dat_multiple.mean(axis=1)) == len(np.mean(dat_multiple.data,
                axis=1))

    # Test std
    assert isinstance(dat_multiple.std(axis=0), Adjacency)
    assert len(dat_multiple.std(axis=0)) == 1
    assert len(dat_multiple.std(axis=1)) == len(np.std(dat_multiple.data,
                axis=1))

    # Test similarity
    assert len(dat_multiple.similarity(
                dat_single.squareform())) == len(dat_multiple)
    assert len(dat_multiple.similarity(dat_single.squareform(),
                metric='pearson')) == len(dat_multiple)
    assert len(dat_multiple.similarity(dat_single.squareform(),
                metric='kendall')) == len(dat_multiple)

    # Test distance
    assert isinstance(dat_multiple.distance(), Adjacency)
    assert dat_multiple.distance().square_shape()[0] == len(dat_multiple)

    # Test ttest
    out = dat_multiple.ttest()
    assert len(out['t']) == 1
    assert len(out['p']) == 1
    assert out['t'].shape()[0] == dat_multiple.shape()[1]
    assert out['p'].shape()[0] == dat_multiple.shape()[1]
    out = dat_multiple.ttest(permutation=True)
    assert len(out['t']) == 1
    assert len(out['p']) == 1
    assert out['t'].shape()[0] == dat_multiple.shape()[1]
    assert out['p'].shape()[0] == dat_multiple.shape()[1]

    # Test Threshold
    assert np.sum(dat_directed.threshold(upper=.8).data == 0) == 10
    assert dat_directed.threshold(upper=.8, binarize=True).data[0]
    assert np.sum(dat_directed.threshold(upper='70%', binarize=True).data) == 5
    assert np.sum(dat_directed.threshold(lower=.4, binarize=True).data) == 6

    # Test to_graph()
    assert isinstance(dat_directed.to_graph(), nx.DiGraph)
    assert isinstance(dat_single.to_graph(), nx.Graph)

    # Test Append
    a = Adjacency()
    a = a.append(dat_single)
    assert a.shape() == dat_single.shape()
    a = a.append(a)
    assert a.shape() == (2, 6)

    n_samples = 3
    b = dat_multiple.bootstrap('mean', n_samples=n_samples)
    assert isinstance(b['Z'], Adjacency)
    b = dat_multiple.bootstrap('std', n_samples=n_samples)
    assert isinstance(b['Z'], Adjacency)

    # Test plot
    f = dat_single.plot()
    assert isinstance(f, plt.Figure)
    f = dat_multiple.plot()
    assert isinstance(f, plt.Figure)

    # Test plot_mds
    f = dat_single.plot_mds()
    assert isinstance(f, plt.Figure)

    # Test Conversion
    np.testing.assert_approx_equal(-1,pearsonr(dat_single.data,dat_single.distance_to_similarity().data)[0],significant=1)
    np.testing.assert_approx_equal(-1,pearsonr(dat_single.distance_to_similarity().data,dat_single.distance_to_similarity().similarity_to_distance().data)[0],significant=1)

    # Test within_cluster_mean
    test_dat = Adjacency(block_diag(np.ones((4,4)),np.ones((4,4))*2,np.ones((4,4))*3),matrix_type='similarity')
    test_labels = np.concatenate([np.ones(4)*x for x in range(1,4)])
    out = test_dat.within_cluster_mean(clusters=test_labels)
    assert np.sum(np.array([1,2,3])-np.array([out[x] for x in out]))==0

    # Test Adjacency Regression
    m1 = block_diag(np.ones((4,4)),np.zeros((4,4)),np.zeros((4,4)))
    m2 = block_diag(np.zeros((4,4)),np.ones((4,4)),np.zeros((4,4)))
    m3 = block_diag(np.zeros((4,4)),np.zeros((4,4)),np.ones((4,4)))
    Y = Adjacency(m1*1+m2*2+m3*3,matrix_type='similarity')
    X = Adjacency([m1,m2,m3],matrix_type='similarity')

    stats = Y.regress(X)
    assert np.allclose(stats['beta'],np.array([1,2,3]))

    # Test Design_Matrix Regression
    n = 10
    d = Adjacency([block_diag(np.ones((4,4))+np.random.randn(4,4)*.1,np.zeros((8,8))) for x in range(n)],
                  matrix_type='similarity')
    X = Design_Matrix(np.ones(n))
    stats = d.regress(X)
    out = stats['beta'].within_cluster_mean(clusters=['Group1']*4 + ['Group2']*8)
    assert np.allclose(np.array([out['Group1'],out['Group2']]),np.array([1,0]), rtol=1e-01)# np.allclose(np.sum(stats['beta']-np.array([1,2,3])),0)

    # # Test stats_label_distance - FAILED - Need to sort this out
    # labels = np.array(['group1','group1','group2','group2'])
    # stats = dat_multiple[0].stats_label_distance(labels)
    # assert np.isclose(stats['group1']['mean'],-1*stats['group2']['mean'])

def test_groupby(tmpdir):
    # Simulate Brain Data
    sim = Simulator()
    r = 10
    sigma = 1
    y = [0, 1]
    n_reps = 3
    output_dir = str(tmpdir)
    sim.create_data(y, sigma, reps=n_reps, output_dir=output_dir)

    s1 = create_sphere([12, 10, -8], radius=r)
    s2 = create_sphere([22, -2, -22], radius=r)
    mask = Brain_Data([s1, s2])

    y = pd.read_csv(os.path.join(str(tmpdir.join('y.csv'))),
                    header=None, index_col=None)
    data = Brain_Data(glob.glob(str(tmpdir.join('data.nii.gz'))), Y=y)
    data.X = pd.DataFrame({'Intercept':np.ones(len(data.Y)),
                        'X1':np.array(data.Y).flatten()}, index=None)

    dat = Groupby(data, mask)

    # Test length
    assert len(dat) == len(mask)

    # Test Index
    assert isinstance(dat[1], Brain_Data)

    # Test apply
    mn = dat.apply('mean')
    assert len(dat) == len(mn)
    # assert mn[0].mean() > mn[1].mean() #JC edit: it seems this check relies on chance from simulated data
    assert mn[1].shape() == np.sum(mask[1].data == 1)
    reg = dat.apply('regress')
    assert len(dat) == len(mn)
    # r = dict([(x,reg[x]['beta'][1]) for x in reg.iterkeys()])

    # Test combine
    combine_mn = dat.combine(mn)
    assert len(combine_mn.shape()) == 1

def test_designmat(tmpdir):

    # Design matrices are specified in terms of sampling frequency
    TR = 2.0
    sampling_freq = 1. / TR

    mat = Design_Matrix(np.random.randint(2,size=(500,4)),columns=['face_A','face_B','house_A','house_B'],sampling_freq=sampling_freq)

    # Basic ops
    matp = mat.add_poly(2)
    assert matp.shape[1] == 7
    assert mat.add_poly(2,include_lower=False).shape[1] == 5

    matpd = matp.add_dct_basis()
    assert matpd.shape[1] == 18

    assert all(matpd.vif() < 2.0)
    assert not all(matpd.vif(exclude_polys=False) < 2.0)

    matc = matpd.clean()
    assert matc.shape[1] == 16

    # Standard convolve
    assert matpd.convolve().shape == matpd.shape
    # Multi-kernal convolve
    hrf = glover_hrf(TR,oversampling=1.)
    assert matpd.convolve(conv_func=np.column_stack([hrf,hrf])).shape[1] == matpd.shape[1] + 4

    matz = mat.zscore(columns = ['face_A','face_B'])
    assert (matz[['house_A','house_B']] == mat[['house_A','house_B']]).all().all()

    # replace data
    assert matpd.replace_data(np.zeros((500,4))).shape == matpd.shape

    #Up and down sampling
    newTR = 1.
    target = 1./newTR
    assert matpd.upsample(target).shape[0] == matpd.shape[0]*2 - target*2
    newTR = 4.
    target = 1./newTR
    assert mat.downsample(target).shape[0] == matpd.shape[0]/2

    # Appending
    mats = matpd.append(matpd)
    assert mats.shape[0] == mat.shape[0] * 2
    # Keep polys separate by default
    assert (mats.shape[1] - 4) == (matpd.shape[1] - 4) * 2
    # Otherwise stack them
    assert matpd.append(matpd,keep_separate=False).shape[1] == matpd.shape[1]
    # Keep a single stimulus column separate
    assert matpd.append(matpd,unique_cols=['face_A']).shape[1] == 33
    # Keep a common stimulus class separate
    assert matpd.append(matpd,unique_cols=['face*']).shape[1] == 34
    # Keep a common stimulus class and a different single stim separate
    assert matpd.append(matpd,unique_cols=['face*','house_A']).shape[1] == 35
    # Keep multiple stimulus class separate
    assert matpd.append(matpd,unique_cols=['face*','house*']).shape[1] == 36

    # Growing a multi-run design matrix; keeping things separate
    num_runs = 4
    all_runs = Design_Matrix(sampling_freq=.5)
    for i in range(num_runs):
        run = Design_Matrix(np.array([
                                [1,0,0,0],
                                [1,0,0,0],
                                [0,0,0,0],
                                [0,1,0,0],
                                [0,1,0,0],
                                [0,0,0,0],
                                [0,0,1,0],
                                [0,0,1,0],
                                [0,0,0,0],
                                [0,0,0,1],
                                [0,0,0,1]
                                ]),
                                sampling_freq = .5,
                                columns=['stim_A','stim_B','cond_C','cond_D']
                                )
        run = run.add_poly(2)
        all_runs = all_runs.append(run,unique_cols=['stim*','cond*'])

    assert all_runs.shape == (44, 28)
