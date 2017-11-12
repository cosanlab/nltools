from __future__ import division

'''
NeuroLearn Data Classes
=======================

Classes to represent various types of data

'''

# Notes:
# Need to figure out how to speed up loading and resampling of data

__all__ = ['Brain_Data',
            'Adjacency',
            'Groupby',
            'Design_Matrix',
            'Design_Matrix_Series']
__author__ = ["Luke Chang"]
__license__ = "MIT"

from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import six
from nltools.utils import (make_cosine_basis,
                            glover_hrf)
from nltools.stats import (pearson,
                           fdr,
                           threshold,
                           fisher_r_to_z,
                           correlation_permutation,
                           one_sample_permutation,
                           two_sample_permutation,
                           downsample,
                           upsample,
                           zscore,
                           )
class Design_Matrix_Series(Series):

    """
    This is a sub-class of pandas series. While not having additional methods
    of it's own required to retain normal slicing functionality for the
    Design_Matrix class, i.e. how slicing is typically handled in pandas.
    All methods should be called on Design_Matrix below.
    """

    @property
    def _constructor(self):
        return Design_Matrix_Series

    @property
    def _constructor_expanddim(self):
        return Design_Matrix


class Design_Matrix(DataFrame):

    """Design_Matrix is a class to represent design matrices with convenience
        functionality for convolution, upsampling and downsampling. It plays
        nicely with Brain_Data and can be used to build an experimental design
        to pass to Brain_Data's X attribute. It is essentially an enhanced
        pandas df, with extra attributes and methods. Methods always return a
        new design matrix instance.

    Args:
        convolved (list, optional): on what columns convolution has been performed; defaults to None
        hasIntercept (bool, optional): whether the design matrix has an
                            intercept column; defaults to False
        sampling_rate (float, optional): sampling rate of each row in seconds (e.g. TR in neuroimaging); defaults to None

    """

    _metadata = ['sampling_rate', 'convolved', 'hasIntercept']

    def __init__(self, *args, **kwargs):

        sampling_rate = kwargs.pop('sampling_rate', None)
        convolved = kwargs.pop('convolved', None)
        hasIntercept = kwargs.pop('hasIntercept', False)
        self.sampling_rate = sampling_rate
        self.convolved = convolved
        self.hasIntercept = hasIntercept

        super(Design_Matrix, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return Design_Matrix

    @property
    def _constructor_sliced(self):
        return Design_Matrix_Series


    def info(self):
        """Print class meta data.

        """
        return '%s.%s(sampling_rate=%s, shape=%s, convolved=%s, hasIntercept=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.sampling_rate,
            self.shape,
            self.convolved,
            self.hasIntercept
            )

    def append(self, df, axis, **kwargs):
        """Method for concatenating another design matrix row or column-wise.
            Can "uniquify" certain columns when appending row-wise, and by
            default will attempt to do that with the intercept.

        Args:
            axis (int): 0 for row-wise (vert-cat), 1 for column-wise (horz-cat)
            separate (bool,optional): whether try and uniquify columns;
                                        defaults to True; only applies
                                        when axis==0
            addIntercept (bool,optional): whether to add intercepts to matrices
                                        before appending; defaults to False;
                                        only applies when axis==0
            uniqueCols (list,optional): what additional columns to try to keep
                                        separated by uniquifying; defaults to
                                        intercept only; only applies when
                                        axis==0

        """
        if axis == 1:
            return self.horzcat(df)
        elif axis == 0:
            return self.vertcat(df, **kwargs)
        else:
            raise ValueError("Axis must be 0 (row) or 1 (column)")


    def horzcat(self, df):
        """Used by .append(). Append another design matrix, column-wise
            (horz cat). Always returns a new design_matrix.

        """
        if self.shape[0] != df.shape[0]:
            raise ValueError("Can't append differently sized design matrices! "
                             "Mat 1 has %s rows and Mat 2 has %s rows."
                            % (self.shape[0]), df.shape[0])
        out = pd.concat([self, df], axis=1)
        out.sampling_rate = self.sampling_rate
        out.convolved = self.convolved
        out.hasIntercept = self.hasIntercept
        return out


    def vertcat(self, df, separate=True, addIntercept=False, uniqueCols=[]):
        """Used by .append(). Append another design matrix row-wise (vert cat).
            Always returns a new design matrix.

        """

        outdf = df.copy()
        assert self.hasIntercept == outdf.hasIntercept, ("Intercepts are "
                "ambigious. Both design matrices should match in whether "
                "they do or don't have intercepts.")

        if addIntercept:
            self['intercept'] = 1
            self.hasIntercept = True
            outdf['intercept'] = 1
            outdf.hasIntercept = True
        if separate:
            if self.hasIntercept:
                uniqueCols += ['intercept']
            idx_1 = []
            idx_2 = []
            for col in uniqueCols:
                # To match substrings within column names, we loop over each
                # element and search for the uniqueCol as a substring within it;
                # first we do to check if the uniqueCol actually occurs in both
                # design matrices, if so then we change it's name before
                # concatenating
                joint = set(self.columns) & set(outdf.columns)
                shared = [elem for elem in joint if col in elem]
                if shared:
                    idx_1 += [i for i, elem in enumerate(self.columns) if col in elem]
                    idx_2 += [i for i, elem in enumerate(outdf.columns) if col in elem]
            aRename = {self.columns[elem]: '0_'+self.columns[elem] for i, elem in enumerate(idx_1)}
            bRename = {outdf.columns[elem]: '1_'+outdf.columns[elem] for i, elem in enumerate(idx_2)}
            if aRename:
                out = self.rename(columns=aRename)
                outdf = outdf.rename(columns=bRename)
                colOrder = []
                #retain original column order as closely as possible
                for colA,colB in zip(out.columns, outdf.columns):
                    colOrder.append(colA)
                    if colA != colB:
                        colOrder.append(colB)
                out = super(Design_Matrix, out).append(outdf, ignore_index=True)
                # out = out.append(outdf,separate=False,axis=0,ignore_index=True).fillna(0)
                out = out[colOrder]
            else:
                raise ValueError("Separate concatentation impossible. None of "
                                "the requested unique columns were found in "
                                "both design_matrices.")
        else:
            out = super(Design_Matrix, self).append(outdf, ignore_index=True)
            # out = self.append(df,separate=False,axis=0,ignore_index=True).fillna(0)

        out.convolved = self.convolved
        out.sampling_rate = self.sampling_rate
        out.hasIntercept = self.hasIntercept

        return out

    def vif(self):
        """Compute variance inflation factor amongst columns of design matrix,
            ignoring the intercept. Much faster that statsmodels and more
            reliable too. Uses the same method as Matlab and R (diagonal
            elements of the inverted correlation matrix).

        Returns:
            vifs (list): list with length == number of columns - intercept

        """
        assert self.shape[1] > 1, "Can't compute vif with only 1 column!"
        if self.hasIntercept:
            idx = [i for i, elem in enumerate(self.columns) if 'intercept' not in elem]
            out = self[self.columns[np.r_[idx]]]
        else:
            out = self[self.columns]

        if any(out.corr() < 0):
            warnings.warn("Correlation matrix has negative values!")

        return np.diag(np.linalg.inv(out.corr()), 0)

    def heatmap(self, figsize=(8, 6), **kwargs):
        """Visualize Design Matrix spm style. Use .plot() for typical pandas
            plotting functionality. Can pass optional keyword args to seaborn
            heatmap.

        """
        fig, ax = plt.subplots(1, figsize=figsize)
        ax = sns.heatmap(self, cmap='gray', cbar=False, ax=ax, **kwargs)
        for _, spine in ax.spines.items():
            spine.set_visible(True)
        for i, label in enumerate(ax.get_yticklabels()):
            if i == 0 or i == self.shape[0]-1:
                label.set_visible(True)
            else:
                label.set_visible(False)
        ax.axhline(linewidth=4, color="k")
        ax.axvline(linewidth=4, color="k")
        ax.axhline(y=self.shape[0], color='k', linewidth=4)
        ax.axvline(x=self.shape[1], color='k', linewidth=4)
        plt.yticks(rotation=0)

    def convolve(self, conv_func='hrf', colNames=None):
        """Perform convolution using an arbitrary function.

        Args:
            conv_func (ndarray or string): either a 1d numpy array containing output of a function that you want to convolve; a samples by kernel 2d array of several kernels to convolve; or th string 'hrf' which defaults to a glover HRF function at the Design_matrix's sampling_freq
            colNames (list): what columns to perform convolution on; defaults
                            to all skipping intercept, and columns containing 'poly' or 'cosine'

        """
        assert self.sampling_rate is not None, "Design_matrix has no sampling_rate set!"

        if colNames is None:
            colNames = [col for col in self.columns if 'intercept' not in col and 'poly' not in col and 'cosine' not in col]
        nonConvolved = [col for col in self.columns if col not in colNames]

        if isinstance(conv_func,six.string_types):
            assert conv_func == 'hrf',"Did you mean 'hrf'? 'hrf' can generate a kernel for you, otherwise custom kernels should be passed in as 1d or 2d arrays."

            assert self.sampling_rate is not None, "Design_matrix sampling rate not set. Can't figure out how to generate HRF!"
            conv_func = glover_hrf(self.sampling_rate, oversampling=1)

        else:
            assert type(conv_func) == np.ndarray, 'Must provide a function for convolution!'

        if len(conv_func.shape) > 1:
            assert conv_func.shape[0] > conv_func.shape[1], '2d conv_func must be formatted as, samples X kernels!'
            conv_mats = []
            for i in range(conv_func.shape[1]):
                c = self[colNames].apply(lambda x: np.convolve(x, conv_func[:,i])[:self.shape[0]])
                c.columns = [str(col)+'_c'+str(i) for col in c.columns]
                conv_mats.append(c)
                out = pd.concat(conv_mats+ [self[nonConvolved]], axis=1)
        else:
            c = self[colNames].apply(lambda x: np.convolve(x, conv_func)[:self.shape[0]])
            c.columns = [str(col)+'_c0' for col in c.columns]
            out = pd.concat([c,self[nonConvolved]], axis=1)

        out.convolved = colNames
        out.sampling_rate = self.sampling_rate
        out.hasIntercept = self.hasIntercept
        return out

    def downsample(self, target,**kwargs):
        """Downsample columns of design matrix. Relies on
            nltools.stats.downsample, but ensures that returned object is a
            design matrix.

        Args:
            target(float): downsampling target, typically in samples not
                            seconds
            kwargs: additional inputs to nltools.stats.downsample

        """
        df = downsample(self, sampling_freq=self.sampling_rate, target=target, **kwargs)

        # convert df to a design matrix
        newMat = Design_Matrix(df, sampling_rate=target,
                               convolved=self.convolved,
                               hasIntercept=self.hasIntercept)
        return newMat

    def upsample(self, target,**kwargs):
        """Upsample columns of design matrix. Relies on
            nltools.stats.upsample, but ensures that returned object is a
            design matrix.

        Args:
            target(float): downsampling target, typically in samples not
                            seconds
            kwargs: additional inputs to nltools.stats.downsample

        """
        df = upsample(self, sampling_freq=self.sampling_rate, target=target, **kwargs)

        # convert df to a design matrix
        newMat = Design_Matrix(df, sampling_rate=target,
                               convolved=self.convolved,
                               hasIntercept=self.hasIntercept)
        return newMat

    def zscore(self, colNames=[]):
        """Z-score specific columns of design matrix. Relies on
            nltools.stats.downsample, but ensures that returned object is a
            design matrix.

        Args:
            colNames (list): columns to z-score; defaults to all columns

        """
        colOrder = self.columns
        if not list(colNames):
            colNames = self.columns
        nonZ = [col for col in self.columns if col not in colNames]
        df = zscore(self[colNames])
        df = pd.concat([df, self[nonZ]], axis=1)
        df = df[colOrder]
        newMat = Design_Matrix(df, sampling_rate=self.sampling_rate,
                               convolved=self.convolved,
                               hasIntercept=self.hasIntercept)

        return newMat

    def addpoly(self, order=0, include_lower=True):
        """Add nth order polynomial terms as columns to design matrix.

        Args:
            order (int): what order terms to add; 0 = constant/intercept
                        (default), 1 = linear, 2 = quadratic, etc
            include_lower: (bool) whether to add lower order terms if order > 0

        """

        #This method is kind of ugly
        polyDict = {}
        if include_lower:
            if order > 0:
                for i in range(0, order+1):
                    if i == 0:
                        if self.hasIntercept:
                            warnings.warn("Design Matrix already has "
                                          "intercept...skipping")
                        else:
                            polyDict['intercept'] = np.repeat(1, self.shape[0])
                    else:
                        polyDict['poly_' + str(i)] = (range(self.shape[0]) - np.mean(range(self.shape[0]))) ** i
            else:
                if self.hasIntercept:
                    raise ValueError("Design Matrix already has intercept!")
                else:
                    polyDict['intercept'] = np.repeat(1, self.shape[0])
        else:
            if order == 0:
                if self.hasIntercept:
                    raise ValueError("Design Matrix already has intercept!")
                else:
                    polyDict['intercept'] = np.repeat(1, self.shape[0])
            else:
                polyDict['poly_'+str(order)] = (range(self.shape[0]) - np.mean(range(self.shape[0])))**order

        toAdd = Design_Matrix(polyDict)
        out = self.append(toAdd, axis=1)
        if 'intercept' in polyDict.keys() or self.hasIntercept:
            out.hasIntercept = True
        return out

    def add_dct_basis(self,duration=180):
        """Adds cosine basis functions to Design_Matrix columns,
        based on spm-style discrete cosine transform for use in
        high-pass filtering.

        Args:
            duration (int): length of filter in seconds

        """
        assert self.sampling_rate is not None, "Design_Matrix has no sampling_rate set!"
        basis_mat = make_cosine_basis(self.shape[0],self.sampling_rate,duration)

        basis_frame = Design_Matrix(basis_mat,
                                    sampling_rate=self.sampling_rate,
                                    hasIntercept=False, convolved=False)

        basis_frame.columns = ['cosine_'+str(i+1) for i in range(basis_frame.shape[1])]

        out = self.append(basis_frame,axis=1)

        return out

def _vif(X, y):
    """
        DEPRECATED
        Helper function to compute variance inflation factor. Unclear whether
        there are errors with this method relative to stats.models. Seems like
        stats.models is sometimes inconsistent with R. R always uses the
        diagonals of the inverted covariance matrix which is what's implemented
        instead of this.

        Args:
            X: (Dataframe) explanatory variables
            y: (Dataframe/Series) outcome variable

    """

    b,resid, _, _ = np.linalg.lstsq(X, y)
    SStot = y.var(ddof=0)*len(y)
    if SStot == 0:
        SStot = .0001  # to prevent divide by 0 errors
    r2 = 1.0 - (resid/SStot)
    if r2 == 1:
        r2 = 0.9999  # to prevent divide by 0 errors
    return (1.0/(1.0-r2))[0]
