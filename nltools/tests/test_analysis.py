from __future__ import division
import os
import numpy as np
import nibabel as nb
import pandas as pd
from nltools.simulator import Simulator
from nltools.analysis import Roc
from nltools.data import Brain_Data
import matplotlib
matplotlib.use('TkAgg')

def test_roc(tmpdir):
    sim = Simulator()

    r = 10
    sigma = .1
    y = [0, 1]
    n_reps = 10
    #     output_dir = str(tmpdir)
    dat = sim.create_data(y, sigma, reps=n_reps, output_dir=None)
    # dat = Brain_Data(data=sim.data, Y=sim.y)

    algorithm = 'svm'
    # output_dir = str(tmpdir)
    # cv = {'type': 'kfolds', 'n_folds': 5, 'subject_id': sim.rep_id}
    extra = {'kernel': 'linear'}

    output = dat.predict(algorithm='svm', plot=False, **extra)

    # Single-Interval
    roc = Roc(input_values=output['yfit_all'], binary_outcome=output['Y'] == 1)
    roc.calculate()
    roc.summary()
    assert roc.accuracy == 1

    # Forced Choice
    binary_outcome = output['Y'] == 1
    forced_choice = list(range(int(len(binary_outcome)/2))) + list(range(int(len(binary_outcome)/2)))
    forced_choice = forced_choice.sort()
    roc_fc = Roc(input_values=output['yfit_all'], binary_outcome=binary_outcome, forced_choice=forced_choice)
    roc_fc.calculate()
    assert roc_fc.accuracy == 1
    assert roc_fc.accuracy == roc_fc.auc == roc_fc.sensitivity == roc_fc.specificity
