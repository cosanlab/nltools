"""
Adjacency Class
===============

Nltools has an additional data structure class for working with two-dimensional
square matrices. This can be helpful when working with similarity/distance
matrices or directed or undirected graphs. Similar to the Brain_Data class,
matrices are vectorized and can store multiple matrices in the same object.
This might reflect different brain regions, subjects, or time. Most of the
methods on the Adjacency class are consistent with those in the Brain_Data
class.

"""

#########################################################################
#  Load Data
# ----------
#
# Similar to the Brain_Data class, Adjacency instances can be initialized by passing in a numpy array or pandas data frame, or a path to a csv file or list of files.  Here we will generate some fake data to demonstrate how to use this class.  In addition to data, you must indicate the type of matrix.  Currently, you can specify `['similarity','distance','directed']`.  Similarity matrices are symmetrical with typically ones along diagonal, Distance matrices are symmetrical with zeros along diagonal, and Directed graph matrices are not symmetrical.  Symmetrical matrices only store the upper triangle. The Adjacency class can also accommodate labels, but does not require them.

from nltools.data import Adjacency
from scipy.linalg import block_diag
import numpy as np

m1 = block_diag(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
m2 = block_diag(np.zeros((4, 4)), np.ones((4, 4)), np.zeros((4, 4)))
m3 = block_diag(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))
noisy = (m1 * 1 + m2 * 2 + m3 * 3) + np.random.randn(12, 12) * 0.1
dat = Adjacency(
    noisy, matrix_type="similarity", labels=["C1"] * 4 + ["C2"] * 4 + ["C3"] * 4
)

#########################################################################
# Basic information about the object can be viewed by simply calling it.

print(dat)

#########################################################################
# Adjacency objects can easily be converted back into two-dimensional matrices with the `.squareform()` method.

dat.squareform()

#########################################################################
# Matrices can viewed as a heatmap using the `.plot()` method.

dat.plot()

#########################################################################
# The mean within a a grouping label can be calculated using the `.cluster_summary()` method.  You must specify a group variable to group the  data.  Here we use the labels.

print(dat.cluster_summary(clusters=dat.labels, summary="within", metric="mean"))

#########################################################################
# Regression
# ----------
#
# Adjacency objects can currently accommodate two different types of regression. Sometimes we might want to decompose an Adjacency matrix from a linear combination of other Adjacency matrices.  Other times we might want to perform a regression at each pixel in a stack of Adjacency matrices. Here we provide an example of each method.  We use the same data we generated above, but attempt to decompose it by each block of data.  We create the design matrix by simply concatenating the matrices we used to create the data object. The regress method returns a dictionary containing all of the relevant information from the regression. Here we show that the model recovers the average weight in each block.

X = Adjacency([m1, m2, m3], matrix_type="similarity")
stats = dat.regress(X)
print(stats["beta"])

#########################################################################
# In addition to decomposing a single adjacency matrix, we can also estimate a model that predicts the variance over each voxel.  This is equivalent to a univariate regression in imaging analyses. Remember that just like in imaging these tests are non-independent and may require correcting for multiple comparisons.  Here we create some data that varies over matrices and identify pixels that follow a particular on-off-on pattern.  We plot the t-values that exceed 2.

from nltools.data import Design_Matrix
import matplotlib.pyplot as plt

data = Adjacency(
    [m1 + np.random.randn(12, 12) * 0.5 for x in range(5)]
    + [np.zeros((12, 12)) + np.random.randn(12, 12) * 0.5 for x in range(5)]
    + [m1 + np.random.randn(12, 12) * 0.5 for x in range(5)]
)

X = Design_Matrix([1] * 5 + [0] * 5 + [1] * 5)
f = X.plot()
f.set_title("Model", fontsize=18)

stats = data.regress(X)
t = stats["t"].plot(vmin=2)
plt.title("Significant Pixels", fontsize=18)

#########################################################################
# Similarity/Distance
# -------------------
#
# We can calculate similarity between two Adjacency matrices using `.similiarity()`.

stats = dat.similarity(m1)
print(stats)

#########################################################################
# We can also calculate the distance between multiple matrices contained within a single Adjacency object. Any distance metric is available from the `sci-kit learn <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html>`_ by specifying the `method` flag. This outputs an Adjacency matrix.  In the example below we see that several matrices are more similar to each other (i.e., when the signal is on).  Remember that the nodes here now represent each matrix from the original distance matrix.

dist = data.distance(metric="correlation")
dist.plot()

#########################################################################
# Similarity matrices can be converted to and from Distance matrices using `.similarity_to_distance()` and `.distance_to_similarity()`.

dist.distance_to_similarity(metric="correlation").plot()

#########################################################################
# Multidimensional Scaling
# ------------------------
#
# We can perform additional analyses on distance matrices such as multidimensional scaling. Here we provide an example to create a 3D multidimensional scaling plot of our data to see if the on and off matrices might naturally group together.

dist = data.distance(metric="correlation")
dist.labels = ["On"] * 5 + ["Off"] * 5 + ["On"] * 5
dist.plot_mds(n_components=3)

#########################################################################
# Graphs
# ------
#
# Adjacency matrices can be cast to networkx objects using `.to_graph()` if the optional dependency is installed. This allows any graph theoretic metrics or plots to be easily calculated from Adjacency objects.

import networkx as nx

dat = Adjacency(m1 + m2 + m3, matrix_type="similarity")
g = dat.to_graph()

print("Degree of each node: %s" % g.degree())

nx.draw_circular(g)
