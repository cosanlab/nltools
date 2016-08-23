"""
============================================================
Example of two level univariate regression on simulated data
============================================================
This example simulates data according to a very simple sketch of brain
imaging data and applies a standard two-level univariate GLM to identify
significant voxels.
"""

__author__ = ["Luke Chang"]
__license__ = "MIT"

print(__doc__)

from time import time

import glob
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from nltools.simulator import Simulator
from nltools.utils import get_resource_path, get_anatomical
from nltools.analysis import Predict, Roc
from nltools.data import Brain_Data
from nltools.stats import threshold
from nltools.mask import create_sphere
import matplotlib.pyplot as plt
import shutil
import tempfile

tmp_dir = os.path.join(tempfile.gettempdir(), str(os.times()[-1]))

###############################################################################
# Create data

tic = time() #Start Timer

sim = Simulator()
r=10
sigma = .5
cor = .8
cov = .6
n_trials = 10
n_subs = 5
s1 = create_sphere([41, 64, 55], radius=r)
sim.create_cov_data(cor, cov, sigma, mask=s1, reps = n_trials, n_sub = n_subs, output_dir = tmp_dir)
print 'Simulate Data: Elapsed: %.2f seconds' % (time() - tic) #Stop timer

###############################################################################
# Load data

tic = time() #Start Timer

y=pd.read_csv(os.path.join(tmp_dir,'y.csv'),header=None,index_col=None).T
dat = Brain_Data(data=os.path.join(tmp_dir,'maskdata_cor0.8_cov0.6_sigma0.5.nii.gz'),Y=y)
dat.X = pd.DataFrame({'Intercept':np.ones(len(dat.Y)),'X1':np.array(dat.Y).flatten()},index=None)
holdout = pd.read_csv(os.path.join(tmp_dir,'rep_id.csv'),header=None,index_col=None).T

print 'Load Data: Elapsed: %.2f seconds' % (time() - tic) #Stop timer


###############################################################################
# Run Regression separately for each subject

tic = time() #Start Timer

start = 0
stop = n_trials
dat.X = pd.DataFrame({'Intercept':np.ones(len(dat.Y)),'X1':np.array(dat.Y).flatten()},index=None)
all = dat.empty()
for i in xrange(n_subs):
    sub_out = dat[start:stop].regress()
    start = start + n_trials
    stop = stop + n_trials
    tmp = sub_out['beta'].empty(data=False)[1]
    all = all.append(tmp)
print 'Regression: Elapsed: %.2f seconds' % (time() - tic) # Stop timer

###############################################################################
# Run One sample t-test

tic = time() #Start Timer

l2 = all.ttest(threshold_dict={'fdr':.05})
l2['thr_t'].plot()
print 'T-Test: Elapsed: %.2f seconds' % (time() - tic) # Stop timer

shutil.rmtree(tmp_dir, ignore_errors=True) # Delete Data

plt.show()
