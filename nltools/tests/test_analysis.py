import os
import numpy as np
import nibabel as nb
import pandas as pd
from nltools import analysis, simulator
from nltools.data import Brain_Data
import matplotlib
matplotlib.use('TkAgg')

def test_roc(tmpdir):
    sim = simulator.Simulator()

    r = 10
    sigma = .1
    y = [0, 1]
    n_reps = 10
    #     output_dir = str(tmpdir)
    sim.create_data(y, sigma, reps=n_reps, output_dir=None)
    dat = Brain_Data(data=sim.data,Y=pd.DataFrame(sim.y))

    algorithm = 'svm'
    # output_dir = str(tmpdir)
    # cv = {'type': 'kfolds', 'n_folds': 5, 'subject_id': sim.rep_id}
    extra = {'kernel': 'linear'}

    output = dat.predict(algorithm='svm', plot=False, **extra)

    # Single-Interval
    roc = analysis.Roc(input_values=output['yfit_all'], binary_outcome=output['Y'] == 1)
    roc.calculate()
    roc.summary()
    assert roc.accuracy == 1

    # Forced Choice
    binary_outcome = output['Y'] == 1
    forced_choice = range(len(binary_outcome)/2) + range(len(binary_outcome)/2)
    forced_choice = forced_choice.sort()
    roc_fc = analysis.Roc(input_values=output['yfit_all'], binary_outcome=binary_outcome, forced_choice=forced_choice)
    roc_fc.calculate()
    assert roc_fc.accuracy == 1
    assert roc_fc.accuracy == roc_fc.auc == roc_fc.sensitivity == roc_fc.specificity

