from __future__ import division

'''
NeuroLearn Design Matrix
========================

Class for working with design matrices.

'''

__author__ = ["Eshin Jolly"]
__license__ = "MIT"

from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import six
from ..external.hrf import glover_hrf
from nltools.stats import (downsample,
                           upsample,
                           zscore,
                           make_cosine_basis
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
        sampling_rate (float): sampling rate of each row in seconds (e.g. TR in neuroimaging)
        convolved (list, optional): on what columns convolution has been performed; defaults to None
        polys (list, optional): list of polynomial terms in design matrix, e.g. intercept, polynomial trends, basis functions, etc; default None


    """

    _metadata = ['sampling_rate', 'convolved', 'polys']

    def __init__(self, *args, **kwargs):

        sampling_rate = kwargs.pop('sampling_rate',None)
        convolved = kwargs.pop('convolved', [])
        polys = kwargs.pop('polys', [])
        self.sampling_rate = sampling_rate
        self.convolved = convolved
        self.polys = polys

        super(Design_Matrix, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return Design_Matrix

    @property
    def _constructor_sliced(self):
        return Design_Matrix_Series

    def _inherit_attributes(self,
                            dm_out,
                            atts=[
                            'sampling_rate',
                            'convolved',
                            'polys']):

        """
        This is helper function that simply ensures that attributes are copied over from an the current Design_Matrix to a new Design_Matrix.

        Args:
            dm_out (Design_Matrix): the new design matrix to copy attributes to
            atts (list; optional): the list of attributes to copy

        Returns:
            dm_out (Design_matrix): new design matrix

        """

        for item in atts:
            setattr(dm_out, item, getattr(self,item))
        return dm_out

    def info(self):
        """Print class meta data.

        """
        return '%s.%s(sampling_rate=%s, shape=%s, convolved=%s, constant_terms=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.sampling_rate,
            self.shape,
            self.convolved,
            self.polys
            )

    def append(self, dm, axis=0, keep_separate = True, addpoly = None, unique_cols = [], include_lower = True,fill_na=0):
        """Method for concatenating another design matrix row or column-wise.
            Can "uniquify" certain columns when appending row-wise, and by
            default will attempt to do that with all polynomial terms (e.g. intercept, polynomial trends).

        Args:
            dm (Design_Matrix or list): design_matrix or list of design_matrices to append
            axis (int): 0 for row-wise (vert-cat), 1 for column-wise (horz-cat); default 0
            keep_separate (bool,optional): whether try and uniquify columns;
                                        defaults to True; only applies
                                        when axis==0
            addpoly (int,optional): what order polynomial terms to add during append, only applied when axis = 0; defaults None;
                                        only applies when axis==0
            unique_cols (list,optional): what additional columns to try to keep
                                        separated by uniquifying, only applies when
                                        axis = 0; defaults to None
            include_lower (bool,optional): whether to also add lower order polynomial terms; only applies when addpoly is not None
            fill_na (str/int/float): if provided will fill NaNs with this value during row-wise appending (when axis = 0) if separate columns are desired; default 0

        """
        if isinstance(dm, list):
            # Check that the first object in a list is a design matrix
            if not isinstance(dm[0],self.__class__):
                raise TypeError("Can only append other Design_Matrix objects!")
            # Check all remaining objects are also design matrices
            if not all([isinstance(elem,dm[0].__class__) for elem in dm]):
                raise TypeError("Each object in list must be a Design_Matrix!")
        elif not isinstance(dm, self.__class__):
            raise TypeError("Can only append other Design_Matrix objects")

        if axis == 1:
            if isinstance(dm,self.__class__):
                if not set(self.columns).isdisjoint(dm.columns):
                    warnings.warn("Duplicate column names detected. Will be repeated.")
            else:
                if any([not set(self.columns).isdisjoint(elem.columns) for elem in dm]):
                    warnings.warn("Duplicate column names detected. Will be repeated.")
            return self.horzcat(dm)
        elif axis == 0:
            return self.vertcat(dm, keep_separate=keep_separate,addpoly=addpoly,unique_cols=unique_cols,include_lower=include_lower,fill_na=fill_na)
        else:
            raise ValueError("Axis must be 0 (row) or 1 (column)")


    def horzcat(self, df):
        """Used by .append(). Append another design matrix, column-wise
            (horz cat). Always returns a new design_matrix.

        """

        if not isinstance(df, list):
            to_append = [df]
        else:
            to_append = df # No need to copy here cause we're not altering df
        if all([elem.shape[0] == self.shape[0] for elem in to_append]):
            out = pd.concat([self] + to_append, axis=1)
        else:
            raise ValueError("All Design Matrices must have the same number of rows!")
        out = self._inherit_attributes(out)
        return out

    def vertcat(self, df, keep_separate, addpoly, unique_cols, include_lower,fill_na):
        """Used by .append(). Append another design matrix row-wise (vert cat).
            Always returns a new design matrix.

        """

        if unique_cols and not keep_separate:
            raise ValueError("Unique columns provided by keep_separate set to False. Set keep_separate to True to separate unique_cols")

        # Convert everything to a list to make things easier
        if not isinstance(df, list):
            to_append = [df]
        else:
            to_append = df[:] # need to make a copy because we're altering df

        if keep_separate:
            if not all([set(self.polys) == set(elem.polys) for elem in to_append]):
                raise ValueError("Design matrices do not match on their polynomial terms (i.e. intercepts, polynomial trends, basis functions). This makes appending with separation ambigious and is not currently supported. Either make sure all constant terms are the same or make sure no Design Matrix has any constant terms and add them during appending with the 'addpoly' and 'unique_cols' arguments")

            orig = self.copy() # Make a copy of the original cause we might alter it

            if addpoly:
                orig = orig.addpoly(addpoly,include_lower)
                for i,d in enumerate(to_append):
                    d = d.addpoly(addpoly,include_lower)
                    to_append[i] = d

            unique_cols += orig.polys
            all_dms = [orig] + to_append
            all_polys = []
            for i,dm in enumerate(all_dms):
                # Figure out what columns we need to relabel
                cols_to_relabel = [col for col in dm.columns if col in unique_cols]
                if cols_to_relabel:
                    # Create a dictionary with the new names, e.g. {'intercept': '0_intercept'}
                    cols_dict = {}
                    # Rename the columns and update the dm
                    for c in cols_to_relabel:
                        cols_dict[c] = str(i) + '_' + c
                    dm = dm.rename(columns=cols_dict)
                    all_dms[i] = dm
                    # Save the new column names to setting the attribute later
                    for v in list(cols_dict.values()):
                        all_polys.append(v)

            out = pd.concat(all_dms,axis=0,ignore_index=True)
            if fill_na:
                out = out.fill_na(fill_na)

            # colOrder = []
            # #retain original column order as closely as possible
            # for colA,colB in zip(out.columns, outdf.columns):
            #     colOrder.append(colA)
            #     if colA != colB:
            #         colOrder.append(colB)
            # out = out[colOrder]
            out.sampling_rate = self.sampling_rate
            out.convolved = self.convolved
            out.polys = all_polys
        else:
            out = pd.concat([self] + to_append,axis=0,ignore_index=True)
            out = self._inherit_attributes(out)
            if addpoly:
                out = out.addpoly(addpoly,include_lower)

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
        if self.has_intercept:
            idx = [i for i, elem in enumerate(self.columns) if 'intercept' not in elem]
            out = self[self.columns[np.r_[idx]]]
        else:
            out = self[self.columns]

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
            conv_func (ndarray or string): either a 1d numpy array containing output of a function that you want to convolve; a samples by kernel 2d array of several kernels to convolve; or th string 'hrf' which defaults to a glover HRF function at the Design_matrix's sampling_rate
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

        out = self._inherit_attributes(out)
        out.convolved = colNames
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
        newMat = self._inherit_attributes(df)
        newMat.sampling_rate = target
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
        newMat = self._inherit_attributes(df)
        newMat.sampling_rate = target
        return newMat

    def zscore(self, columns=[]):
        """Z-score specific columns of design matrix. Relies on
            nltools.stats.downsample, but ensures that returned object is a
            design matrix.

        Args:
            columns (list): columns to z-score; defaults to all columns

        """
        colOrder = self.columns
        if not list(columns):
            columns = self.columns
        nonZ = [col for col in self.columns if col not in columns]
        df = zscore(self[columns])
        df = pd.concat([df, self[nonZ]], axis=1)
        df = df[colOrder]
        newMat = self._inherit_attributes(df)

        return newMat

    def addpoly(self, order=0, include_lower=True):
        """Add nth order polynomial terms as columns to design matrix.

        Args:
            order (int): what order terms to add; 0 = constant/intercept
                        (default), 1 = linear, 2 = quadratic, etc
            include_lower: (bool) whether to add lower order terms if order > 0

        """

        if order < 0:
            raise ValueError("Order must be 0 or greater")

        polyDict = {}

        if order == 0 and 'intercept' in self.polys:
            raise ValueError("Design Matrix already has intercept")
        elif 'poly_'+str(order) in self.polys:
            raise ValueError("Design Matrix already has {}th order polynomial".format(order))

        if include_lower:
            for i in range(0, order+1):
                if i == 0:
                    if 'intercept' in self.polys:                            warnings.warn("Design Matrix already has "
                                      "intercept...skipping")
                    else:
                        polyDict['intercept'] = np.repeat(1, self.shape[0])
                else:
                    if 'poly_'+str(i) in self.polys:
                        warnings.warn("Design Matrix already has {}th order polynomial...skipping".format(i))
                    else:
                        polyDict['poly_' + str(i)] = (range(self.shape[0]) - np.mean(range(self.shape[0]))) ** i
        else:
            if order == 0:
                polyDict['intercept'] = np.repeat(1, self.shape[0])
            else:
                polyDict['poly_'+str(order)] = (range(self.shape[0]) - np.mean(range(self.shape[0])))**order

        toAdd = Design_Matrix(polyDict,sampling_rate=self.sampling_rate)
        out = self.append(toAdd, axis=1)
        if out.polys:
            new_polys = out.polys + list(polyDict.keys())
            out.polys = new_polys
        else:
            out.polys = list(polyDict.keys())
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
                                    sampling_rate=self.sampling_rate)

        basis_frame.columns = ['cosine_'+str(i+1) for i in range(basis_frame.shape[1])]

        out = self.append(basis_frame,axis=1)
        if out.polys:
            new_polys = out.polys + list(basis_frame.columns)
            out.polys = new_polys
        else:
            out.polys = list(basis_frame.columns)
        return out
