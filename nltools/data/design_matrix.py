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
from scipy.special import legendre
import six
from ..external.hrf import glover_hrf
from nltools.stats import (downsample,
                           upsample,
                           zscore,
                           make_cosine_basis
                           )
from nltools.utils import AmbiguityError


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

    """Design_Matrix is a class to represent design matrices with special methods for data processing (e.g. convolution, upsampling, downsampling) and also intelligent and flexible and intelligent appending (e.g. auto-matically keep certain columns or polynomial terms separated during concatentation). It plays nicely with Brain_Data and can be used to build an experimental design to pass to Brain_Data's X attribute. It is essentially an enhanced pandas df, with extra attributes and methods. Methods always return a new design matrix instance (copy). Column names are always string types.

    Args:
        sampling_freq (float): sampling rate of each row in hertz; To covert seconds to hertz (e.g. in the case of TRs for neuroimaging) using hertz = 1 / TR
        convolved (list, optional): on what columns convolution has been performed; defaults to None
        polys (list, optional): list of polynomial terms in design matrix, e.g. intercept, polynomial trends, basis functions, etc; default None

    """

    _metadata = ['sampling_freq', 'convolved', 'polys', 'multi']

    def __init__(self, *args, **kwargs):

        sampling_freq = kwargs.pop('sampling_freq', None)
        convolved = kwargs.pop('convolved', [])
        polys = kwargs.pop('polys', [])
        self.sampling_freq = sampling_freq
        self.convolved = convolved
        self.polys = polys
        self.multi = False

        super(Design_Matrix, self).__init__(*args, **kwargs)
        # Ensure that column names are string types to all methods work
        if not self.empty:
            self.columns = [str(elem) for elem in self.columns]

    @property
    def _constructor(self):
        return Design_Matrix

    @property
    def _constructor_sliced(self):
        return Design_Matrix_Series

    def _inherit_attributes(self,
                            dm_out,
                            atts=[
                                'sampling_freq',
                                'convolved',
                                'polys',
                                'multi']):
        """
        This is helper function that simply ensures that attributes are copied over from  the current Design_Matrix to a new Design_Matrix.

        Args:
            dm_out (Design_Matrix): the new design matrix to copy attributes to
            atts (list; optional): the list of attributes to copy

        Returns:
            dm_out (Design_matrix): new design matrix

        """

        for item in atts:
            setattr(dm_out, item, getattr(self, item))
        return dm_out

    def _sort_cols(self):
        """
        This is a helper function that tries to ensure that columns of a Design Matrix are sorted according to: a) those not separated during append operations, b) those separated during append operations, c) polynomials. Called primarily during vertical concatentation and cleaning.
        """
        data_cols = [elem for elem in self.columns if not elem.split('_')[0].isdigit() and elem not in self.polys]
        separated_cols = [elem for elem in self.columns if elem.split('_')[0].isdigit() and elem not in self.polys]
        return self[data_cols + separated_cols + self.polys]

    def details(self):
        """Print class meta data.

        """
        return '%s.%s(sampling_freq=%s (hz), shape=%s, multi=%s, convolved=%s, polynomials=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.sampling_freq,
            self.shape,
            self.multi,
            self.convolved,
            self.polys
            )

    def append(self, dm, axis=0, keep_separate=True, unique_cols=[], fill_na=0, verbose=False):
        """Method for concatenating another design matrix row or column-wise. When concatenating row-wise, has the ability to keep certain columns separated if they exist in multiple design matrices (e.g. keeping separate intercepts for multiple runs). This is on by default and will automatically separate out polynomial columns (i.e. anything added with the `add_poly` or `add_dct_basis` methods). Additional columns can be separate by run using the `unique_cols` parameter. Can also add new polynomial terms during vertical concatentation (when axis == 0). This will by default create new polynomial terms separately for each design matrix

        Args:
            dm (Design_Matrix or list): design_matrix or list of design_matrices to append
            axis (int): 0 for row-wise (vert-cat), 1 for column-wise (horz-cat); default 0
            keep_separate (bool,optional): whether try and uniquify columns;
                                        defaults to True; only applies
                                        when axis==0
            unique_cols (list,optional): what additional columns to try to keep
                                        separated by uniquifying, only applies when
                                        axis = 0; defaults to None
            fill_na (str/int/float): if provided will fill NaNs with this value during row-wise appending (when axis = 0) if separate columns are desired; default 0
            verbose (bool): print messages during append about how polynomials are going to be separated

        """
        if not isinstance(dm, list):
            to_append = [dm]
        else:
            to_append = dm[:]

        # Check all items to be appended are Design Matrices and have the same sampling rate
        if not all([isinstance(elem, self.__class__) for elem in to_append]):
            raise TypeError("Each object to be appended must be a Design_Matrix!")
        if not all([elem.sampling_freq == self.sampling_freq for elem in to_append]):
            raise ValueError("All Design Matrices must have the same sampling frequency!")

        if axis == 1:
            if any([not set(self.columns).isdisjoint(elem.columns) for elem in to_append]):
                print("Duplicate column names detected. Will be repeated.")
            return self._horzcat(to_append, fill_na=fill_na)

        elif axis == 0:
            return self._vertcat(to_append, keep_separate=keep_separate, unique_cols=unique_cols, fill_na=fill_na, verbose=verbose)

        else:
            raise ValueError("Axis must be 0 (row) or 1 (column)")

    def _horzcat(self, to_append, fill_na):
        """Used by .append(). Append another design matrix, column-wise
            (horz cat). Always returns a new design_matrix.

        """

        if all([elem.shape[0] == self.shape[0] for elem in to_append]):
            out = pd.concat([self] + to_append, axis=1)
            out = self._inherit_attributes(out)
            out.polys = self.polys[:]
            for elem in to_append:
                out.polys += elem.polys
            if fill_na is not None:
                out = out.fillna(fill_na)
        else:
            raise ValueError("All Design Matrices must have the same number of rows!")
        return out

    def _vertcat(self, df, keep_separate, unique_cols, fill_na, verbose):
        """Used by .append(). Append another design matrix row-wise (vert cat).
            Always returns a new design matrix.

        """

        # make a copy of the dms to append
        to_append = df[:]
        orig = self.copy()  # Make a copy of the original cause we might alter it

        # In order to append while keeping things separated we're going to create a new list of dataframes to append with renamed columns
        modify_to_append = []
        all_polys = []
        cols_to_separate = []
        all_separated = []

        if len(unique_cols):
            if not keep_separate:
                raise ValueError("unique_cols provided but keep_separate set to False. Set keep_separate to True to separate unique_cols")

            # 1) Make sure unique_cols are in original Design Matrix
            if not self.empty:
                to_rename = {}
                unique_count = []
                for u in unique_cols:
                    if u.endswith('*'):
                        searchstr = u.split('*')[0]
                    elif u.startswith('*'):
                        searchstr = u.split('*')[1]
                    else:
                        searchstr = u
                    if not any([searchstr in elem for elem in self.columns]):
                        raise ValueError("'{}' not present in any column name of original Design Matrix".format(searchstr))
            # 2) Prepend them with a 0_ if this dm has never been appended to be for otherwise grab their current prepended index are and start a unique_cols counter
                    else:
                        for c in self.columns:
                            if searchstr in c:
                                if self.multi and c[0].isdigit():
                                    count = c.split('_')[0]
                                    unique_count.append(int(count))
                                else:
                                    new_name = '0_' + c
                                    all_separated.append(new_name)
                                    to_rename[c] = new_name
                                    all_separated.append(new_name)
                    cols_to_separate.append(searchstr)

                if to_rename:
                    orig = orig.rename(columns=to_rename)
                    max_unique_count = 0
                else:
                    max_unique_count = np.array(unique_count).max()

        # 3) Handle several different cases:
        # a) original has no polys, dms to append do
        # b) original has no polys, dms to append dont
        # c) original has polys, dms to append do
        # d) original has polys, dms to append dont
        # Within each of these also keep a counter, update, and check for unique cols if needed
        # This unique_col checking code is uglyly repeated in each conditional branch of a-d, but differs in subtle ways; probably could be cleaned up in a refactor
        if keep_separate:
            if not len(self.polys):
                # Self no polys; append has polys.
                if any([len(elem.polys) for elem in to_append]):
                    if verbose:
                        print("Keep separate requested but original Design Matrix has no polynomial terms but matrices to be appended do. Inherting appended Design Matrices' polynomials...")
                    for i, dm in enumerate(to_append):
                        for p in dm.polys:
                            all_polys.append(p)

                        # Handle renaming additional unique cols to keep separate
                        if cols_to_separate:
                            if verbose:
                                print("Unique cols requested. Trying to keep {} separated".format(cols_to_separate))
                            to_rename = {}
                            data_cols = dm.drop(dm.polys, axis=1).columns
                            print(data_cols)
                            for u in cols_to_separate:
                                for c in data_cols:
                                    if u in c:
                                        if dm.multi:
                                            count = int(c.split('_')[0])
                                            name = '_'.join(c.split('_')[1:])
                                            count += max_unique_count + 1
                                            new_name = str(count) + '_' + name
                                            to_rename[c] = new_name
                                        else:
                                            new_name = str(max_unique_count + 1) + '_' + c
                                            to_rename[c] = new_name
                                        all_separated.append(new_name)
                            modify_to_append.append(dm.rename(columns=to_rename))
                            max_unique_count += 1
                        else:
                            modify_to_append.append(dm)
                else:
                    # Self no polys; append no polys
                    if verbose:
                        print("Keep separate requested but neither original Design Matrix nor matrices to be appended have any polynomial terms Ignoring...")
                    # Handle renaming additional unique cols to keep separate
                    for i, dm in enumerate(to_append):
                        if cols_to_separate:
                            if verbose:
                                print("Unique cols requested. Trying to keep {} separated".format(cols_to_separate))
                            to_rename = {}
                            data_cols = dm.drop(dm.polys, axis=1).columns
                            for u in cols_to_separate:
                                for c in data_cols:
                                    if u in c:
                                        if dm.multi:
                                            count = int(c.split('_')[0])
                                            name = '_'.join(c.split('_')[1:])
                                            count += max_unique_count + 1
                                            new_name = str(count) + '_' + name
                                            to_rename[c] = new_name
                                        else:
                                            new_name = str(max_unique_count + 1) + '_' + c
                                            to_rename[c] = new_name
                                        all_separated.append(new_name)
                            modify_to_append.append(dm.rename(columns=to_rename))
                            max_unique_count += 1
                        else:
                            modify_to_append.append(dm)
            else:
                # Self has polys; append has polys
                if any([len(elem.polys) for elem in to_append]):
                    if verbose:
                        print("Keep separate requested and both original Design Matrix and matrices to be appended have polynomial terms. Separating...")
                    # Get the unique polynomials that currently exist
                    # [name, count/None, isRoot]
                    current_polys = []
                    for p in self.polys:
                        if p.count('_') == 2:
                            isRoot = False
                            pSplit = p.split('_')
                            pName = '_'.join(pSplit[1:])
                            pCount = int(pSplit[0])
                        else:
                            isRoot = True
                            pName = p
                            pCount = 0
                        current_polys.append([pName, pCount, isRoot])

                    # Mixed type numpy array to make things a little easier
                    current_polys = pd.DataFrame(current_polys).values

                    # If current polynomials dont begin with a prepended numerical identifier, created one, e.g. 0_poly_1
                    if any(current_polys[:, 2]):
                        renamed_polys = {}
                        for i in range(current_polys.shape[0]):
                            renamed_polys[current_polys[i, 0]] = str(current_polys[i, 1]) + '_' + current_polys[i, 0]
                        orig = orig.rename(columns=renamed_polys)
                        all_polys += list(renamed_polys.values())
                    else:
                        all_polys += self.polys

                    current_poly_max = current_polys[:, 1].max()

                    for i, dm in enumerate(to_append):
                        to_rename = {}
                        for p in dm.polys:
                            if p.count('_') == 2:
                                pSplit = p.split('_')
                                pName = '_'.join(pSplit[1:])
                                pCount = int(pSplit[0]) + current_poly_max + 1
                            else:
                                pName = p
                                pCount = current_poly_max + 1
                            to_rename[p] = str(pCount) + '_' + pName
                        temp_dm = dm.rename(columns=to_rename)
                        current_poly_max += 1
                        all_polys += list(to_rename.values())

                        # Handle renaming additional unique cols to keep separate
                        if cols_to_separate:
                            if verbose:
                                print("Unique cols requested. Trying to keep {} separated".format(cols_to_separate))
                            to_rename = {}
                            data_cols = dm.drop(dm.polys, axis=1).columns
                            for u in cols_to_separate:
                                for c in data_cols:
                                    if u in c:
                                        if dm.multi:
                                            count = int(c.split('_')[0])
                                            name = '_'.join(c.split('_')[1:])
                                            count += max_unique_count + 1
                                            new_name = str(count) + '_' + name
                                            to_rename[c] = new_name
                                        else:
                                            new_name = str(max_unique_count + 1) + '_' + c
                                            to_rename[c] = new_name
                                        all_separated.append(new_name)

                            # Combine renamed polynomials and renamed uniqu_cols
                            modify_to_append.append(temp_dm.rename(columns=to_rename))
                            max_unique_count += 1
                        else:
                            modify_to_append.append(temp_dm)
                else:
                    # Self has polys; append no polys
                    if verbose:
                        print("Keep separate requested but only original Design Matrix has polynomial terms. Retaining original Design Matrix's polynomials only...")
                    all_polys += self.polys

                    # Handle renaming additional unique cols to keep separate
                    if cols_to_separate:
                        if verbose:
                            print("Unique cols requested. Trying to keep {} separated".format(cols_to_separate))
                        for i, dm in enumerate(to_append):
                            to_rename = {}
                            data_cols = dm.drop(dm.polys, axis=1).columns
                            for u in cols_to_separate:
                                for c in data_cols:
                                    if u in c:
                                        if dm.multi:
                                            count = int(c.split('_')[0])
                                            name = '_'.join(c.split('_')[1:])
                                            count += max_unique_count + 1
                                            new_name = str(count) + '_' + name
                                            to_rename[c] = new_name
                                        else:
                                            new_name = str(max_unique_count + 1) + '_' + c
                                            to_rename[c] = new_name
                                        all_separated.append(new_name)
                        modify_to_append.append(dm.rename(to_rename))
                        max_unique_count += 1
                    else:
                        modify_to_append.append(dm)

        # Combine original dm with the updated/renamed dms to be appended
        all_dms = [orig] + modify_to_append

        out = pd.concat(all_dms, axis=0, ignore_index=True, sort=True)

        if fill_na is not None:
            out = out.fillna(fill_na)

        out.sampling_freq = self.sampling_freq
        out.convolved = self.convolved
        out.multi = True
        out.polys = all_polys

        return out._sort_cols()

    def vif(self, exclude_polys=True):
        """Compute variance inflation factor amongst columns of design matrix,
            ignoring polynomial terms. Much faster that statsmodels and more
            reliable too. Uses the same method as Matlab and R (diagonal
            elements of the inverted correlation matrix).

        Returns:
            vifs (list): list with length == number of columns - intercept
            exclude_polys (bool): whether to skip checking of polynomial terms (i.e. intercept, trends, basis functions); default True

        """
        if self.shape[1] <= 1:
            raise ValueError("Can't compute vif with only 1 column!")
        if self.polys and exclude_polys:
            out = self.drop(self.polys, axis=1)
        else:
            # Always drop intercept before computing VIF
            intercepts = [elem for elem in self.columns if 'poly_0' in str(elem)]
            out = self.drop(intercepts, axis=1)
        try:
            return np.diag(np.linalg.inv(out.corr()), 0)
        except np.linalg.LinAlgError:
            print("ERROR: Cannot compute vifs! Design Matrix is singular because it has some perfectly correlated or duplicated columns. Using .clean() method may help.")

    def heatmap(self, figsize=(8, 6), **kwargs):
        """Visualize Design Matrix spm style. Use .plot() for typical pandas
            plotting functionality. Can pass optional keyword args to seaborn
            heatmap.

        """
        cmap = kwargs.pop('cmap', 'gray')
        fig, ax = plt.subplots(1, figsize=figsize)
        ax = sns.heatmap(self, cmap=cmap, cbar=False, ax=ax, **kwargs)
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
            conv_func (ndarray or string): either a 1d numpy array containing output of a function that you want to convolve; a samples by kernel 2d array of several kernels to convolve; or the string 'hrf' which defaults to a glover HRF function at the Design_matrix's sampling_freq
            columns (list): what columns to perform convolution on; defaults
                            to all non-polynomial columns

        """
        if self.sampling_freq is None:
            raise ValueError("Design_matrix has no sampling_freq set!")

        if columns is None:
            columns = [col for col in self.columns if col not in self.polys]
        nonConvolved = [col for col in self.columns if col not in columns]

        if isinstance(conv_func, np.ndarray):
            if len(conv_func.shape) > 2:
                raise ValueError("2d conv_func must be formatted as samplex X kernals!")
        elif isinstance(conv_func, six.string_types):
            if conv_func != 'hrf':
                raise ValueError("Did you mean 'hrf'? 'hrf' can generate a kernel for you, otherwise custom kernels should be passed in as 1d or 2d arrays.")
            conv_func = glover_hrf(1. / self.sampling_freq, oversampling=1.)

        else:
            raise TypeError("conv_func must be a 1d or 2d numpy array organized as samples x kernels, or the string 'hrf' for the canonical glover hrf")

        if len(conv_func.shape) > 1:
            conv_mats = []
            for i in range(conv_func.shape[1]):
                c = self[columns].apply(lambda x: np.convolve(x, conv_func[:, i])[:self.shape[0]])
                c.columns = [str(col)+'_c'+str(i) for col in c.columns]
                conv_mats.append(c)
                out = pd.concat(conv_mats + [self[nonConvolved]], axis=1)
        else:
            c = self[columns].apply(lambda x: np.convolve(x, conv_func)[:self.shape[0]])
            c.columns = [str(col)+'_c0' for col in c.columns]
            out = pd.concat([c, self[nonConvolved]], axis=1)

        out = self._inherit_attributes(out)
        out.convolved = columns
        return out

    def downsample(self, target, **kwargs):
        """Downsample columns of design matrix. Relies on
            nltools.stats.downsample, but ensures that returned object is a
            design matrix.

        Args:
            target(float): desired frequency in hz
            kwargs: additional inputs to nltools.stats.downsample

        """
        if target > self.sampling_freq:
            raise ValueError("Target must be longer than current sampling rate")

        df = Design_Matrix(downsample(self, sampling_freq=self.sampling_freq, target=target, target_type='hz', **kwargs))

        # convert df to a design matrix
        newMat = self._inherit_attributes(df)
        newMat.sampling_freq = target
        return newMat

    def upsample(self, target, **kwargs):
        """Upsample columns of design matrix. Relies on
            nltools.stats.upsample, but ensures that returned object is a
            design matrix.

        Args:
            target(float): desired frequence in hz
            kwargs: additional inputs to nltools.stats.downsample

        """
        if target < self.sampling_freq:
            raise ValueError("Target must be shorter than current sampling rate")

        df = Design_Matrix(upsample(self, sampling_freq=self.sampling_freq, target=target, target_type='hz', **kwargs))

        # convert df to a design matrix
        newMat = self._inherit_attributes(df)
        newMat.sampling_freq = target
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
        """Add nth order Legendre polynomial terms as columns to design matrix. Good for adding constant/intercept to model (order = 0) and accounting for slow-frequency nuisance artifacts e.g. linear, quadratic, etc drifts. Care is recommended when using this with `.add_dct_basis()` as some columns will be highly correlated.

        Args:
            order (int): what order terms to add; 0 = constant/intercept
                        (default), 1 = linear, 2 = quadratic, etc
            include_lower: (bool) whether to add lower order terms if order > 0

        """
        if order < 0:
            raise ValueError("Order must be 0 or greater")

        if self.polys:
            if any([elem.count('_') == 2 for elem in self.polys]):
                raise AmbiguityError("It appears that this Design Matrix contains polynomial terms that were kept seperate from a previous append operation. This makes it ambiguous for adding polynomials terms. Try calling .add_poly() on each separate Design Matrix before appending them instead.")

        polyDict = {}
        # Normal/canonical legendre polynomials on the range -1,1 but with size defined by number of observations; keeps all polynomials on similar scales (i.e. big polys don't blow up) and betas are better behaved
        norm_order = np.linspace(-1, 1, self.shape[0])

        if 'poly_'+str(order) in self.polys:
            print("Design Matrix already has {}th order polynomial...skipping".format(order))
            return self

        if include_lower:
            for i in range(0, order+1):
                if 'poly_'+str(i) in self.polys:
                    print("Design Matrix already has {}th order polynomial...skipping".format(i))
                else:
                    polyDict['poly_' + str(i)] = legendre(i)(norm_order)
        else:
            polyDict['poly_' + str(order)] = legendre(order)(norm_order)

        toAdd = Design_Matrix(polyDict, sampling_freq=self.sampling_freq)
        out = self.append(toAdd, axis=1)
        if out.polys:
            new_polys = out.polys + list(polyDict.keys())
            out.polys = new_polys
        else:
            out.polys = list(polyDict.keys())
        return out

    def add_dct_basis(self, duration=180, drop=0):
        """Adds unit scaled cosine basis functions to Design_Matrix columns,
        based on spm-style discrete cosine transform for use in
        high-pass filtering. Does not add intercept/constant. Care is recommended if using this along with `.add_poly()`, as some columns will be highly-correlated.

        Args:
            duration (int): length of filter in seconds
            drop (int): index of which early/slow bases to drop if any; will always drop constant (i.e. intercept) like SPM. Unlike SPM, retains first basis (i.e. linear/sigmoidal). Will cumulatively drop bases up to and inclusive of index provided (e.g. 2, drops bases 1 and 2); default None

        """
        if self.sampling_freq is None:
            raise ValueError("Design_Matrix has no sampling_freq set!")

        if self.polys:
            if any([elem.count('_') == 2 and 'cosine' in elem for elem in self.polys]):
                raise AmbiguityError("It appears that this Design Matrix contains cosine bases that were kept seperate from a previous append operation. This makes it ambiguous for adding polynomials terms. Try calling .add_dct_basis() on each separate Design Matrix before appending them instead.")

        basis_mat = make_cosine_basis(self.shape[0], 1./self.sampling_freq, duration, drop=drop)

        basis_frame = Design_Matrix(basis_mat,
                                    sampling_freq=self.sampling_freq, columns=[str(elem) for elem in range(basis_mat.shape[1])])

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
            out = self.append(basis_frame, axis=1)
            new_polys = out.polys + list(basis_frame.columns)
            out.polys = new_polys
            return out

    def replace_data(self, data, column_names=None):
        """Convenient method to replace all data in Design_Matrix with new data while keeping attributes and polynomial columns untouched.

        Args:
            columns_names (list): list of columns names for new data

        """

        if isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame) or isinstance(data, dict):
            if data.shape[0] == self.shape[0]:
                out = Design_Matrix(data, columns=column_names)
                polys = self[self.polys]
                out = pd.concat([out, polys], axis=1)
                out = self._inherit_attributes(out)
                return out
            else:
                raise ValueError("New data cannot change the number of rows")
        else:
            raise TypeError("New data must be numpy array, pandas DataFrame or python dictionary type")

    def clean(self, fill_na=0, exclude_polys=False, thresh=.95, verbose=True):
        """
        Method to fill NaNs in Design Matrix and remove duplicate columns based on data values, NOT names. Columns are dropped if they are correlated >= the requested threshold (default = .95). In this case, only the first instance of that column will be retained and all others will be dropped.

        Args:
            fill_na (str/int/float): value to fill NaNs with set to None to retain NaNs; default 0
            exclude_polys (bool): whether to skip checking of polynomial terms (i.e. intercept, trends, basis functions); default False
            thresh (float): correlation threshold to use to drop redundant columns; default .95
            verbose (bool): print what column names were dropped; default True

        """

        # Temporarily turn off warnings for correlations
        old_settings = np.seterr(all='ignore')
        if fill_na is not None:
            out = self.fillna(fill_na)

        if exclude_polys:
            data_cols = [c for c in self.columns if c not in self.polys]
            out = out[data_cols]

        keep = []
        remove = []
        for i, c in out.iteritems():
            for j, c2 in out.iteritems():
                if i != j:
                    r = np.abs(pearsonr(c, c2)[0])
                    if (r >= thresh) and (j not in keep) and (j not in remove):
                        if verbose:
                            print("{} and {} correlated at {} which is >= threshold of {}. Dropping {}".format(i, j, np.round(r, 2), thresh, j))
                        keep.append(i)
                        remove.append(j)
        if remove:
            out = out.drop(remove, axis=1)
            out.polys = [elem for elem in out.polys if elem not in remove]
            out = out._sort_cols()
        else:
            print("Dropping columns not needed...skipping")
        np.seterr(**old_settings)
        return out
