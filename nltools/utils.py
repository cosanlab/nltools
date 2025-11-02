"""
NeuroLearn Utilities
====================

handy utilities.

"""

__all__ = [
    "get_resource_path",
    "set_algorithm",
    "attempt_to_import",
    "all_same",
    "concatenate",
    "_bootstrap_apply_func",
    "set_decomposition_algorithm",
    "get_mni_from_img_resolution",
    "to_h5",
    "load_brain_data_h5",
]

from os.path import dirname, join, sep as pathsep
import nibabel as nib
import importlib
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
import collections
from types import GeneratorType
from h5py import File as h5File
from nltools.prefs import MNI_Template


def to_h5(obj, file_name, obj_type="brain_data", h5_compression="gzip"):
    """Save BrainData or Adjacency objects to HDF5 files.

    Uses a combination of pandas and h5py to save objects to h5 files.

    Args:
        obj: Object to save (BrainData or Adjacency).
        file_name: Path to save file to.
        obj_type: Type of object ('brain_data' or 'adjacency').
        h5_compression: Compression type for h5py datasets.
    """
    if obj_type not in ["brain_data", "adjacency"]:
        raise TypeError("obj_type must be one of 'brain_data' or 'adjacency'")

    if obj_type == "brain_data":
        # Note: X and Y attributes removed in v0.6.0
        # Store empty DataFrames for backward compatibility
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="tables")
            warnings.filterwarnings("ignore", message=".*performance.*", module="tables")
            with pd.HDFStore(file_name, "w") as f:
                # Check if obj has these deprecated attributes for backward compatibility
                if hasattr(obj, "X"):
                    f["X"] = obj.X
                else:
                    f["X"] = pd.DataFrame()
                if hasattr(obj, "Y"):
                    f["Y"] = obj.Y
                else:
                    f["Y"] = pd.DataFrame()

        with h5File(file_name, "a") as f:
            f.create_dataset("data", data=obj.data, compression=h5_compression)
            f.create_dataset(
                "mask_affine", data=obj.mask.affine, compression=h5_compression
            )
            f.create_dataset(
                "mask_data", data=obj.mask.get_fdata(), compression=h5_compression
            )
            f.create_dataset("mask_file_name", data=obj.mask.get_filename())
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="tables")
            warnings.filterwarnings("ignore", message=".*performance.*", module="tables")
            with pd.HDFStore(file_name, "w") as f:
                f["Y"] = obj.Y

        with h5File(file_name, "a") as f:
            f.create_dataset("data", data=obj.data, compression=h5_compression)
            f.create_dataset("matrix_type", data=obj.matrix_type)
            f.create_dataset("issymmetric", data=obj.issymmetric)
            f.create_dataset("labels", data=obj.labels)
            f.create_dataset("is_single_matrix", data=obj.is_single_matrix)


def load_brain_data_h5(file_path, mask=None):
    """Load BrainData from HDF5 file.

    Handles both modern and legacy (pre-0.4.8) HDF5 formats.

    Args:
        file_path: Path to HDF5 file.
        mask: Optional mask to use. If None, loads mask from file if available.

    Returns:
        dict: Dictionary containing loaded data, X, Y, and optionally mask info.
    """

    result = {}

    try:
        # Try modern format first
        with pd.HDFStore(file_path, "r") as f:
            result["X"] = f["X"]
            result["Y"] = f["Y"]

        with h5File(file_path, "r") as f:
            result["data"] = np.array(f["data"])

            # Handle mask loading
            if mask is None and "mask_data" in f:
                # Load mask from file
                result["mask"] = nib.Nifti1Image(
                    np.array(f["mask_data"]),
                    affine=np.array(f["mask_affine"]),
                    file_map={
                        "image": nib.FileHolder(
                            filename=f["mask_file_name"][()].decode()
                        )
                    },
                )
                result["load_mask"] = True
            else:
                result["load_mask"] = False

    except Exception:
        # Fall back to legacy format
        result = _load_legacy_brain_data_h5(file_path, mask)
        result["legacy_format"] = True

    return result


def _load_legacy_brain_data_h5(file_path, mask=None):
    """Load BrainData from legacy HDF5 format (pre-0.4.8).

    Args:
        file_path: Path to HDF5 file.
        mask: Optional mask to use.

    Returns:
        dict: Dictionary containing loaded data, X, Y, and optionally mask info.
    """
    # Import here to avoid circular import
    tables_mod = attempt_to_import("tables")
    if tables_mod is None:
        raise ImportError("tables package required for legacy h5 format")

    result = {}

    with tables_mod.open_file(file_path, mode="r") as f:
        # Load data
        result["data"] = np.array(f.root["data"])

        # Load X DataFrame
        if len(list(f.root["X_columns"])):
            result["X"] = pd.DataFrame(
                np.array(f.root["X"]).squeeze(),
                columns=[
                    e.decode("utf-8") if isinstance(e, bytes) else e
                    for e in np.array(f.root["X_columns"])
                ],
                index=[
                    e.decode("utf-8") if isinstance(e, bytes) else e
                    for e in np.array(f.root["X_index"])
                ],
            )
        else:
            result["X"] = pd.DataFrame()

        # Load Y DataFrame
        if len(list(f.root["Y_columns"])):
            result["Y"] = pd.DataFrame(
                np.array(f.root["Y"]).squeeze(),
                columns=[
                    e.decode("utf-8") if isinstance(e, bytes) else e
                    for e in np.array(f.root["Y_columns"])
                ],
                index=[
                    e.decode("utf-8") if isinstance(e, bytes) else e
                    for e in np.array(f.root["Y_index"])
                ],
            )
        else:
            result["Y"] = pd.DataFrame()

        # Handle mask loading
        if mask is None and "mask_data" in f.root:
            filename = (
                f.root["mask_file_name"]
                if "mask_file_name" in f.root
                else "mask.nii.gz"  # Default filename
            )
            result["mask"] = nib.Nifti1Image(
                np.array(f.root["mask_data"]),
                affine=np.array(f.root["mask_affine"]),
                file_map={"image": nib.FileHolder(filename=filename)},
            )
            result["load_mask"] = True
        else:
            result["load_mask"] = False

    return result


def get_resource_path():
    """Get path to nltools resource directory."""
    return join(dirname(__file__), "resources") + pathsep


def get_anatomical():
    """Get nltools default anatomical image.
    DEPRECATED. Use MNI_Template.plot from nltools.prefs instead.
    """
    from nltools.prefs import MNI_Template

    return nib.load(MNI_Template.plot)


def get_mni_from_img_resolution(brain, img_type="plot"):
    """
    Get the path to the MNI anatomical image that matches the resolution of a BrainData instance.

    This function determines the resolution of the input BrainData and returns the appropriate
    MNI template image path from the current MNI_Template settings, adjusting only the resolution
    while keeping the same template variant.

    Args:
        brain: BrainData instance
        img_type: 'plot' for T1 image or 'brain' for brain-extracted image

    Returns:
        file_path: path to MNI image with matching resolution
    """

    if img_type not in ["plot", "brain"]:
        raise ValueError("img_type must be 'plot' or 'brain' ")

    # Get resolution from the brain data
    res_array = np.abs(np.diag(brain.nifti_masker.affine_)[:3])
    voxel_dims = np.unique(abs(res_array))
    if len(voxel_dims) != 1:
        raise ValueError(
            "Voxels are not isometric and cannot be visualized in standard space"
        )

    # Determine resolution in mm
    resolution = int(voxel_dims[0])

    # Check if this resolution is supported for the current template
    if resolution not in MNI_Template._supported_combinations.get(
        MNI_Template.template, []
    ):
        # If not supported, return the current template's image
        # This handles cases where data resolution doesn't match available templates
        if img_type == "brain":
            return MNI_Template.brain
        else:
            return MNI_Template.plot

    # Build path with matching resolution
    from os.path import join, dirname

    base_path = join(
        dirname(MNI_Template.mask).rsplit("/niftis/", 1)[0],
        "niftis",
        MNI_Template.template,
    )
    res_str = f"{resolution}mm"

    if img_type == "brain":
        return join(base_path, f"MNI152_{res_str}_brain.nii.gz")
    else:
        return join(base_path, f"MNI152_{res_str}_T1.nii.gz")


def set_algorithm(algorithm, *args, **kwargs):
    """Setup the algorithm to use in subsequent prediction analyses.

    Args:
        algorithm: The prediction algorithm to use. Either a string or an
                    (uninitialized) scikit-learn prediction object. If string,
                    must be one of 'svm','svr', linear','logistic','lasso',
                    'lassopcr','lassoCV','ridge','ridgeCV','ridgeClassifier',
                    'randomforest', or 'randomforestClassifier'
        kwargs: Additional keyword arguments to pass onto the scikit-learn
                clustering object.

    Returns:
        predictor_settings: dictionary of settings for prediction

    """

    # NOTE: function currently located here instead of analysis.py to avoid circular imports

    predictor_settings = {}
    predictor_settings["algorithm"] = algorithm

    def load_class(import_string):
        class_data = import_string.split(".")
        module_path = ".".join(class_data[:-1])
        class_str = class_data[-1]
        module = importlib.import_module(module_path)
        return getattr(module, class_str)

    algs_classify = {
        "svm": "sklearn.svm.SVC",
        "logistic": "sklearn.linear_model.LogisticRegression",
        "ridgeClassifier": "sklearn.linear_model.RidgeClassifier",
        "ridgeClassifierCV": "sklearn.linear_model.RidgeClassifierCV",
        "randomforestClassifier": "sklearn.ensemble.RandomForestClassifier",
    }
    algs_predict = {
        "svr": "sklearn.svm.SVR",
        "linear": "sklearn.linear_model.LinearRegression",
        "lasso": "sklearn.linear_model.Lasso",
        "lassoCV": "sklearn.linear_model.LassoCV",
        "ridge": "sklearn.linear_model.Ridge",
        "ridgeCV": "sklearn.linear_model.RidgeCV",
        "randomforest": "sklearn.ensemble.RandomForest",
    }

    if algorithm in algs_classify.keys():
        predictor_settings["prediction_type"] = "classification"
        alg = load_class(algs_classify[algorithm])
        predictor_settings["predictor"] = alg(*args, **kwargs)
    elif algorithm in algs_predict:
        predictor_settings["prediction_type"] = "prediction"
        alg = load_class(algs_predict[algorithm])
        predictor_settings["predictor"] = alg(*args, **kwargs)
    elif algorithm == "lassopcr":
        predictor_settings["prediction_type"] = "prediction"
        from sklearn.linear_model import Lasso
        from sklearn.decomposition import PCA

        predictor_settings["_lasso"] = Lasso()
        predictor_settings["_pca"] = PCA()
        predictor_settings["predictor"] = Pipeline(
            steps=[
                ("pca", predictor_settings["_pca"]),
                ("lasso", predictor_settings["_lasso"]),
            ]
        )
    elif algorithm == "pcr":
        predictor_settings["prediction_type"] = "prediction"
        from sklearn.linear_model import LinearRegression
        from sklearn.decomposition import PCA

        predictor_settings["_regress"] = LinearRegression()
        predictor_settings["_pca"] = PCA()
        predictor_settings["predictor"] = Pipeline(
            steps=[
                ("pca", predictor_settings["_pca"]),
                ("regress", predictor_settings["_regress"]),
            ]
        )
    else:
        raise ValueError(
            """Invalid prediction/classification algorithm name.
            Valid options are 'svm','svr', 'linear', 'logistic', 'lasso',
            'lassopcr','lassoCV','ridge','ridgeCV','ridgeClassifier',
            'randomforest', or 'randomforestClassifier'."""
        )

    return predictor_settings


def set_decomposition_algorithm(algorithm, n_components=None, *args, **kwargs):
    """Setup the algorithm to use in subsequent decomposition analyses.

    Args:
        algorithm: The decomposition algorithm to use. Either a string or an
                    (uninitialized) scikit-learn decomposition object.
                    If string must be one of 'pca','nnmf', ica','fa',
                    'dictionary', 'kernelpca'.
        kwargs: Additional keyword arguments to pass onto the scikit-learn
                clustering object.

    Returns:
        predictor_settings: dictionary of settings for prediction

    """

    # NOTE: function currently located here instead of analysis.py to avoid circular imports

    def load_class(import_string):
        class_data = import_string.split(".")
        module_path = ".".join(class_data[:-1])
        class_str = class_data[-1]
        module = importlib.import_module(module_path)
        return getattr(module, class_str)

    algs = {
        "pca": "sklearn.decomposition.PCA",
        "ica": "sklearn.decomposition.FastICA",
        "nnmf": "sklearn.decomposition.NMF",
        "fa": "sklearn.decomposition.FactorAnalysis",
        "dictionary": "sklearn.decomposition.DictionaryLearning",
        "kernelpca": "sklearn.decomposition.KernelPCA",
    }

    if algorithm in algs.keys():
        alg = load_class(algs[algorithm])
        alg = alg(n_components, *args, **kwargs)
    else:
        raise ValueError(
            """Invalid prediction/classification algorithm name.
            Valid options are 'pca','ica', 'nnmf', 'fa'"""
        )
    return alg


def isiterable(obj):
    """Returns True if the object is one of allowable iterable types."""
    return isinstance(obj, (list, tuple, GeneratorType))


module_names = {}
Dependency = collections.namedtuple("Dependency", "package value")


def attempt_to_import(dependency, name=None, fromlist=None):
    if name is None:
        name = dependency
    try:
        mod = __import__(dependency, fromlist=fromlist)
    except ImportError:
        mod = None
    module_names[name] = Dependency(dependency, mod)
    return mod


def all_same(items):
    return np.all(x == items[0] for x in items)


def concatenate(data):
    """Concatenate a list of BrainData() or Adjacency() objects"""

    if not isinstance(data, list):
        raise ValueError("Make sure you are passing a list of objects.")

    if all([isinstance(x, data[0].__class__) for x in data]):
        # Temporarily Removing this for circular imports (LC)
        # if not isinstance(data[0], (BrainData, Adjacency)):
        #     raise ValueError('Make sure you are passing a list of BrainData'
        #                     ' or Adjacency objects.')

        out = data[0].__class__()
        for i in data:
            out = out.append(i)
    else:
        raise ValueError("Make sure all objects in the list are the same type.")
    return out


def _bootstrap_apply_func(data, function, random_state=None, *args, **kwargs):
    """Bootstrap helper function. Sample with replacement and apply function"""
    random_state = check_random_state(random_state)
    data_row_id = range(data.shape[0])
    new_dat = data[
        random_state.choice(data_row_id, size=len(data_row_id), replace=True)
    ]
    return getattr(new_dat, function)(*args, **kwargs)


def check_square_numpy_matrix(data):
    """Helper function to make sure matrix is square and numpy array"""

    from nltools.data import Adjacency

    if isinstance(data, Adjacency):
        data = data.squareform()
    elif isinstance(data, pd.DataFrame):
        data = data.values
    else:
        data = np.array(data)

    if len(data.shape) != 2:
        try:
            data = squareform(data)
        except ValueError:
            raise ValueError(
                "Array does not contain the correct number of elements to be square"
            )
    return data


def check_brain_data(data, mask=None):
    """Check if data is a BrainData Instance."""
    from nltools.data import BrainData

    if not isinstance(data, BrainData):
        if isinstance(data, nib.Nifti1Image):
            data = BrainData(data, mask=mask)
        else:
            raise ValueError("Make sure data is a BrainData instance.")
    else:
        if mask is not None:
            data = data.apply_mask(mask)
    return data


def check_brain_data_is_single(data):
    """Logical test if BrainData instance is a single image

    Args:
        data: brain data

    Returns:
        (bool)

    """
    data = check_brain_data(data)
    if len(data.shape) > 1:
        return False
    else:
        return True


def _roi_func(brain, roi, algorithm, cv_dict, **kwargs):
    """BrainData.predict_multi() helper function"""
    return brain.apply_mask(roi).predict(
        algorithm=algorithm, cv_dict=cv_dict, plot=False, **kwargs
    )


class AmbiguityError(Exception):
    pass


def generate_jitter(n_trials, mean_time=5, min_time=2, max_time=12, atol=0.2):
    """Generate jitter from exponential distribution with constraints

    Draws from exponential distribution until the distribution satisfies the constraints:
    np.abs(np.mean(min_time > data < max_time) - mean_time) <= atol

    Args:
        n_trials: (int) number of trials to generate jitter
        mean_time: (float) desired mean of distribution
        min_time: (float) desired min of distribution
        max_time: (float) desired max of distribution
        atol: (float) precision of deviation from mean

    Returns:
        data: (np.array) jitter for each trial

    """

    def generate_data(n_trials, scale=5, min_time=2, max_time=12):
        data = []
        i = 0
        while i < n_trials:
            datam = np.random.exponential(scale=5)
            if (datam > min_time) & (datam < max_time):
                data.append(datam)
                i += 1
        return data

    mean_diff = False
    while ~mean_diff:
        data = generate_data(n_trials, min_time=min_time, max_time=max_time)
        mean_diff = np.isclose(np.mean(data), mean_time, rtol=0, atol=atol)
    return data
