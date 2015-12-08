[![Build Status](https://api.travis-ci.org/ljchang/neurolearn.png)](https://travis-ci.org/ljchang/neurolearn/)

neurolearn
==========
Python toolbox for analyzing neuroimaging data.  It is based off of Tor Wager's object oriented matlab <a href=http://wagerlab.colorado.edu/tools>canlab core tools</a> and relies heavily on <a href = http://nilearn.github.io>nilearn</a> and <a href=http://scikit-learn.org/stable/index.html>scikit learn</a>

<h3>Current Tools</h3>
<ul>
<li>Predict: apply various classification and prediction algorithms to 4D dataset</li>
<li>apply_mask: apply 3D weight map to 4D dataset</li>
<li>Roc: perform ROC analysis</li>
</ul>

<h3>Installation</h3>
<ol>
<li>Clone github repository</li>
<li>python setup.py install</li>
</ol>

<h3>Documentation</h3>
<p>
Current Documentation can be found at <a href=http://neurolearn.readthedocs.org/en/latest/>readthedocs</a>.  Please see the ipython notebook examples for walkthroughs of how to use most of the toolbox.
<br><br>
Here is a <a href=https://github.com/ljchang/neurolearn/blob/master/scripts/NLTools_Brain_Data_Class_Tutorial.ipynb>notebook</a> with a detailed overview of how to use the main Brain_Data class.  We also have a <a href=https://github.com/ljchang/neurolearn/blob/master/scripts/Chang_ML_fMRI_Tutorial.ipynb>notebook</a> containing other analysis methods such as prediction and ROI curves (note it is now recommended to use the prediction Brain_Data method).
</p>
