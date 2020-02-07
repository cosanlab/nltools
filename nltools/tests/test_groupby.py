import numpy as np
from nltools.data import Brain_Data


def test_length(sim_groupby):
    assert len(sim_groupby) == len(sim_groupby.mask)


def test_index(sim_groupby):
    assert isinstance(sim_groupby[1], Brain_Data)


def test_apply(sim_groupby):
    mn = sim_groupby.apply("mean")
    assert len(sim_groupby) == len(mn)
    assert mn[1].shape() == np.sum(sim_groupby.mask[1].data == 1)
    # reg = sim_groupby.apply("regress")
    assert len(sim_groupby) == len(mn)


def test_combine(sim_groupby):
    mn = sim_groupby.apply("mean")
    combine_mn = sim_groupby.combine(mn)
    assert len(combine_mn.shape()) == 1
