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
from scipy.stats import pearsonr
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

    def details(self):
        """Print class meta data.

        """
        return '%s.%s(sampling_rate=%s, shape=%s, convolved=%s, polynomials=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.sampling_rate,
            self.shape,
            self.convolved,
            self.polys
            )

    def append(self, dm, axis=0, keep_separate = True, add_poly = None, add_dct_basis = None, unique_cols = [], include_lower = True,fill_na=0):
        """Method for concatenating another design matrix row or column-wise.
            Can "uniquify" certain columns when appending row-wise, and by
            default will attempt to do that with all polynomial terms (e.g. intercept, polynomial trends). Can also add new polynomial terms during vertical concatentation (when axis == 0). This will by default create new polynomial terms separately for each design matrix

        Args:
            dm (Design_Matrix or list): design_matrix or list of design_matrices to append
            axis (int): 0 for row-wise (vert-cat), 1 for column-wise (horz-cat); default 0
            keep_separate (bool,optional): whether try and uniquify columns;
                                        defaults to True; only applies
                                        when axis==0
            add_poly (int,optional): what order polynomial terms to add during append, only applied when axis = 0; default None
            add_dct_basis (int,optional): add discrete cosine bassi function during append, only applied when axis = 0; default None
                                        only applies when axis==0
            unique_cols (list,optional): what additional columns to try to keep
                                        separated by uniquifying, only applies when
                                        axis = 0; defaults to None
            include_lower (bool,optional): whether to also add lower order polynomial terms; only applies when add_poly is not None
            fill_na (str/int/float): if provided will fill NaNs with this value during row-wise appending (when axis = 0) if separate columns are desired; default 0

        """
        if not isinstance(dm, list):
            to_append = [dm]
        else:
            to_append = dm[:]

        # Check all items to be appended are Design Matrices and have the same sampling rate
        if not all([isinstance(elem,self.__class__) for elem in to_append]):
            raise TypeError("Each object in list must be a Design_Matrix!")
        if not all([elem.sampling_rate == self.sampling_rate for elem in to_append]):
            raise ValueError("All Design Matrices must have the same sampling rate!")

        if axis == 1:
            if any([not set(self.columns).isdisjoint(elem.columns) for elem in to_append]):
                print("Duplicate column names detected. Will be repeated.")
            if add_poly or unique_cols:
                print("add_poly and unique_cols only apply when axis=0...ignoring")
            return self._horzcat(to_append)

        elif axis == 0:
            return self._vertcat(to_append, keep_separate=keep_separate,add_poly=add_poly,add_dct_basis=add_dct_basis,unique_cols=unique_cols,include_lower=include_lower,fill_na=fill_na)

        else:
            raise ValueError("Axis must be 0 (row) or 1 (column)")


    def _horzcat(self, to_append):
        """Used by .append(). Append another design matrix, column-wise
            (horz cat). Always returns a new design_matrix.

        """

        if all([elem.shape[0] == self.shape[0] for elem in to_append]):
            out = pd.concat([self] + to_append, axis=1)
            out = self._inherit_attributes(out)
            out.polys = self.polys[:]
            for elem in to_append:
                out.polys += elem.polys
        else:
            raise ValueError("All Design Matrices must have the same number of rows!")
        return out

    def _vertcat(self, df, keep_separate, add_poly, add_dct_basis, unique_cols, include_lower,fill_na):
        """Used by .append(). Append another design matrix row-wise (vert cat).
            Always returns a new design matrix.

        """

        if unique_cols:
            if not keep_separate:
                raise ValueError("unique_cols provided but keep_separate set to False. Set keep_separate to True to separate unique_cols")

        to_append = df[:] # need to make a copy because we're altering df

        if keep_separate:
            if not all([set(self.polys) == set(elem.polys) for elem in to_append]):
                raise ValueError("Design matrices do not match on their polynomial terms (i.e. intercepts, polynomial trends, basis functions). This makes appending with separation ambigious and is not currently supported. Either make sure all constant terms are the same or make sure no Design Matrix has any constant terms and add them during appending with the 'add_poly' and 'unique_cols' arguments")

            orig = self.copy() # Make a copy of the original cause we might alter it

            if add_poly:
                orig = orig.add_poly(add_poly,include_lower)
                for i,d in enumerate(to_append):
                    d = d.add_poly(add_poly,include_lower)
                    to_append[i] = d

            if add_dct_basis:
                orig = orig.add_dct_basis(add_dct_basis)
                for i,d in enumerate(to_append):
                    d = d.add_dct_basis(add_dct_basis)
                    to_append[i] = d

            all_cols = unique_cols + orig.polys
            all_dms = [orig] + to_append
            all_polys = []
            is_data = []
            for i,dm in enumerate(all_dms):
                # Figure out what columns we need to relabel
                cols_to_relabel = [col for col in dm.columns if col in all_cols]
                if cols_to_relabel:
                    # Create a dictionary with the new names, e.g. {'intercept': '0_intercept'}
                    cols_dict = {}
                    # Rename the columns and update the dm
                    for c in cols_to_relabel:
                        cols_dict[c] = str(i) + '_' + c
                        if c not in unique_cols:
                            all_polys.append(cols_dict[c])
                    dm = dm.rename(columns=cols_dict)
                    all_dms[i] = dm

            out = pd.concat(all_dms,axis=0,ignore_index=True)
            if fill_na is not None:
                out = out.fillna(fill_na)

            out.sampling_rate = self.sampling_rate
            out.convolved = self.convolved
            out.polys = all_polys
            data_cols = [elem for elem in out.columns if elem not in out.polys]
            out = out[data_cols + out.polys]
        else:
            out = pd.concat([self] + to_append,axis=0,ignore_index=True)
            out = self._inherit_attributes(out)
            if add_poly:
                out = out.add_poly(add_poly,include_lower)

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
        if self.polys:
            out = self.drop(self.polys,axis=1)
        else:
            out = self[self.columns]

        try:
            return np.diag(np.linalg.inv(out.corr()), 0)
        except np.linalg.LinAlgError:
            print("ERROR: Cannot compute vifs! Design Matrix is singular because it has some perfectly correlated or duplicated columns. Using .clean() method may help.")


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

    def convolve(self, conv_func='hrf', columns=None):
        """Perform convolution using an arbitrary function.

        Args:
            conv_func (ndarray or string): either a 1d numpy array containing output of a function that you want to convolve; a samples by kernel 2d array of several kernels to convolve; or th string 'hrf' which defaults to a glover HRF function at the Design_matrix's sampling_rate
            columns (list): what columns to perform convolution on; defaults
                            to all skipping intercept, and columns containing 'poly' or 'cosine'

        """
        assert self.sampling_rate is not None, "Design_matrix has no sampling_rate set!"

        if columns is None:
            columns = [col for col in self.columns if 'intercept' not in col and 'poly' not in col and 'cosine' not in col]
        nonConvolved = [col for col in self.columns if col not in columns]

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
                c = self[columns].apply(lambda x: np.convolve(x, conv_func[:,i])[:self.shape[0]])
                c.columns = [str(col)+'_c'+str(i) for col in c.columns]
                conv_mats.append(c)
                out = pd.concat(conv_mats+ [self[nonConvolved]], axis=1)
        else:
            c = self[columns].apply(lambda x: np.convolve(x, conv_func)[:self.shape[0]])
            c.columns = [str(col)+'_c0' for col in c.columns]
            out = pd.concat([c,self[nonConvolved]], axis=1)

        out = self._inherit_attributes(out)
        out.convolved = columns
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
        if target < self.sampling_rate:
            raise ValueError("Target must be longer than current sampling rate")

        df = Design_Matrix(downsample(self, sampling_freq=1./self.sampling_rate, target=target,target_type='seconds', **kwargs))

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
        if target > self.sampling_rate:
            raise ValueError("Target must be shorter than current sampling rate")

        df = Design_Matrix(upsample(self, sampling_freq=1./self.sampling_rate, target=target, target_type='seconds',**kwargs))

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

    def add_poly(self, order=0, include_lower=True):
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
            print("Design Matrix already has intercept...skipping")
            return self
        elif 'poly_'+str(order) in self.polys:
            print("Design Matrix already has {}th order polynomial...skipping".format(order))
            return self

        if include_lower:
            for i in range(0, order+1):
                if i == 0:
                    if 'intercept' in self.polys:                            print("Design Matrix already has intercept...skipping")
                    else:
                        polyDict['intercept'] = np.repeat(1, self.shape[0])
                else:
                    if 'poly_'+str(i) in self.polys:
                        print("Design Matrix already has {}th order polynomial...skipping".format(i))
                    else:
                        # Unit scale polynomial terms so they don't blow up
                        vals = np.arange(self.shape[0])
                        vals = (vals - np.mean(vals)) / np.std(vals)
                        polyDict['poly_' + str(i)] = vals ** i
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

        if self.polys:
            # Only add those we don't already have
            basis_to_add = [b for b in basis_frame.columns if b not in self.polys]
        else:
            basis_to_add = list(basis_frame.columns)
        if not basis_to_add:
            print("All basis functions already exist...skipping")
            return self
        else:
            if len(basis_to_add) != len(basis_frame.columns):
                print("Some basis functions already exist...skipping")
            basis_frame = basis_frame[basis_to_add]
            out = self.append(basis_frame,axis=1)
            new_polys = out.polys + list(basis_frame.columns)
            out.polys = new_polys
            return out

    def replace_data(self,data,column_names=None):
        """Convenient method to replace all data in Design_Matrix with new data while keeping attributes and polynomial columns untouched.

        Args:
            columns_names (list): list of columns names for new data

        """

        if isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame) or isinstance(data, dict):
            if data.shape[0] == self.shape[0]:
                out = Design_Matrix(data,columns=column_names)
                polys = self[self.polys]
                out = pd.concat([out,polys],axis=1)
                out = self._inherit_attributes(out)
                return out
            else:
                raise ValueError("New data cannot change the number of rows")
        else:
            raise TypeError("New data must be numpy array, pandas DataFrame or python dictionary type")

    def clean(self,fill_na=0,exclude_polys=False,verbose=True):
        """
        Method to fill NaNs in Design Matrix and remove duplicate columns based on data values, NOT names. Columns are dropped if they cause the Design Matrix to become singular i.e. are perfectly correlated. In this case, only the first instance of that column will be retained and all others will be dropped.

        Args:
            fill_na (str/int/float): value to fill NaNs with set to None to retain NaNs; default 0
            exclude_polys (bool): whether to skip checking of polynomial terms (i.e. intercept, trends, basis functions); default False
            verbose (bool): print what column names were dropped; default True

        """

        # Temporarily turn off warnings for correlations
        old_settings = np.seterr(all='ignore')
        if fill_na is not None:
            out = self.fillna(fill_na)

        if exclude_polys:
            data_cols = [c for c in self.columns if c not in self.polys]
            out = out[data_cols]

        keep = []; remove = []
        for i, c in out.iteritems():
           for j, c2 in out.iteritems():
               if i != j:
                   r = pearsonr(c,c2)[0]
                   if (r > 0.99) and (j not in keep) and (j not in remove):
                       keep.append(i)
                       remove.append(j)
        if remove:
            out = out.drop(remove, axis=1)
        else:
            print("Dropping columns not needed...skipping")
        if verbose:
            print("Dropping columns: ", remove)
        np.seterr(**old_settings)
        return out
