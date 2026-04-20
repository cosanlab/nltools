"""
BrainData prediction functions.

Standalone functions extracted from BrainData class methods for timeseries
prediction (encoding models) and MVPA decoding (pattern classification).
"""

import warnings

import numpy as np

from .utils import shallow_copy


def predict(
    bd,
    X=None,
    y=None,
    method="whole_brain",
    estimator="svm",
    cv=5,
    groups=None,
    roi_mask=None,
    radius_mm=10.0,
    scoring="accuracy",
    standardize=True,
    n_jobs=-1,
    progress_bar=False,
):
    """Generate predictions using fitted model OR classify patterns (MVPA).

    This method supports two prediction modes determined by which parameter
    is provided:

    1. **Timeseries prediction** (X provided): Use fitted ridge model to
       predict voxel responses for new feature data.

    2. **MVPA decoding** (y provided): Train a classifier to predict labels
       from brain patterns using cross-validation.

    Args:
        bd: BrainData instance.
        X (array-like, optional): Features for timeseries prediction, shape
            (n_samples, n_features). If None and y is None, uses training
            data from fit().
        y (array-like, optional): Labels for MVPA decoding, shape (n_samples,).
            If provided, performs pattern classification instead of
            timeseries prediction.

        MVPA-specific parameters (only used when y is provided):

        method (str): Decoding method - 'whole_brain', 'searchlight', or 'roi'.
        estimator (str or sklearn estimator): Classifier to use. Can be:
            - 'svm': LinearSVC (default)
            - 'logistic': LogisticRegression
            - 'ridge': RidgeClassifier
            - 'lda': LinearDiscriminantAnalysis
            - Any sklearn-compatible estimator with fit/predict
        cv (int or sklearn CV splitter): Cross-validation specification.
            Int for k-fold or sklearn CV object.
        groups (array-like, optional): Group labels for CV (e.g., run IDs
            for leave-one-run-out).
        roi_mask (Nifti1Image or str, optional): Atlas/parcellation for
            ROI-based decoding.
        radius_mm (float): searchlight radius in mm. Default: 10.0.
        scoring (str): Metric for evaluation ('accuracy',
            'balanced_accuracy', 'roc_auc').
        standardize (bool): Z-score features before classification.
            Default: True.
        n_jobs (int): Number of parallel jobs for searchlight (-1 = all cores).
        progress_bar (bool): Show progress bar for searchlight.

    Returns:
        BrainData: For timeseries prediction, shape (n_samples, n_voxels).
            For MVPA, shape (1, n_voxels) with accuracy per voxel/ROI.

    Raises:
        ValueError: If both X and y are provided.
        ValueError: If fit() has not been called (for timeseries mode).

    Examples:
        >>> # Timeseries prediction (encoding model)
        >>> brain_data.fit(model='ridge', X=features)
        >>> predictions = brain_data.predict(X=new_features)

        >>> # MVPA decoding (pattern classification)
        >>> # brain_data.data has shape (n_trials, n_voxels)
        >>> accuracy = brain_data.predict(y=labels, method='searchlight')
        >>> print(accuracy.shape)  # (1, n_voxels)
    """
    # Validate mutually exclusive modes
    if X is not None and y is not None:
        raise ValueError(
            "Cannot specify both X and y. Use X for timeseries prediction "
            "or y for MVPA decoding."
        )

    # Dispatch to appropriate mode
    if y is not None:
        return predict_mvpa(
            bd,
            y=y,
            method=method,
            estimator=estimator,
            cv=cv,
            groups=groups,
            roi_mask=roi_mask,
            radius_mm=radius_mm,
            scoring=scoring,
            standardize=standardize,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
        )
    else:
        return predict_timeseries(bd, X=X)


def predict_timeseries(bd, X=None):
    """Generate timeseries predictions using fitted model.

    Internal function for encoding model prediction.

    Args:
        bd: BrainData instance.
        X (array-like, optional): Features to predict on. If None, uses
            training data.

    Returns:
        BrainData with predicted timeseries.
    """
    from nltools.data import BrainData

    # Check model is fitted
    if not hasattr(bd, "model_"):
        raise ValueError(
            "Must call fit() before predict(). "
            "Example: brain_data.fit(model='ridge', X=features)"
        )

    if not bd.model_.is_fitted_:
        raise ValueError("Model is not fitted")

    # Use training data if X not provided
    using_training_data = X is None
    if using_training_data:
        if not hasattr(bd, "X_"):
            raise ValueError(
                "No training data stored. This should not happen - "
                "please report this as a bug."
            )
        X = bd.X_

    # Validate X
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")

    # Validate number of features (handle Ridge and Glm differently)
    if hasattr(bd.model_, "n_features_in_"):
        # Ridge model has n_features_in_
        if X.shape[1] != bd.model_.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with "
                f"{bd.model_.n_features_in_} features"
            )
    else:
        # Glm model - check against design matrices
        if hasattr(bd.model_, "design_matrices_") and bd.model_.design_matrices_:
            expected_features = bd.model_.design_matrices_[0].shape[1]
            if X.shape[1] != expected_features:
                raise ValueError(
                    f"X has {X.shape[1]} features, but model was fitted with "
                    f"{expected_features} features"
                )

    # Generate predictions
    from nltools.models import Glm

    if isinstance(bd.model_, Glm):
        # For GLM, check if using training data or new data
        if using_training_data:
            # Using training data - get fitted values
            y_pred_list = bd.model_.predict()  # Returns list of nifti images
            # Convert to array
            y_pred = BrainData(y_pred_list, mask=bd.mask).data
        else:
            # New design matrix - not yet implemented in Glm
            raise NotImplementedError(
                "Prediction with new design matrix not yet implemented for GLM. "
                "Use predict() without arguments to get fitted values."
            )
    else:
        # Ridge and other models
        y_pred = bd.model_.predict(X)

    # Wrap in BrainData
    predictions = shallow_copy(bd)
    predictions.data = y_pred

    return predictions


def predict_mvpa(
    bd,
    y,
    method="whole_brain",
    estimator="svm",
    cv=5,
    groups=None,
    roi_mask=None,
    radius_mm=10.0,
    scoring="accuracy",
    standardize=True,
    n_jobs=-1,
    progress_bar=False,
):
    """Perform MVPA decoding using cross-validation.

    Internal function for pattern classification.

    Args:
        bd: BrainData instance.
        y (array-like): Labels to predict, shape (n_samples,).
        method (str): 'whole_brain', 'searchlight', or 'roi'.
        estimator (str or sklearn estimator): Classifier (string shortcut
            or sklearn estimator).
        cv (int or sklearn CV splitter): Cross-validation specification.
        groups (array-like, optional): Group labels for CV.
        roi_mask (Nifti1Image or str, optional): Atlas for ROI-based decoding.
        radius_mm (float): searchlight radius in mm.
        scoring (str): Scoring metric.
        standardize (bool): Whether to z-score features.
        n_jobs (int): Parallel jobs for searchlight.
        progress_bar (bool): Show progress bar.

    Returns:
        BrainData with accuracy values.
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Validate method
    valid_methods = {"whole_brain", "searchlight", "roi"}
    if method not in valid_methods:
        raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}")

    # Emit deprecation warning for whole_brain method suggesting fluent API
    if method == "whole_brain":
        warnings.warn(
            "predict(y=labels, cv=k) is deprecated for whole-brain MVPA. "
            "Use the fluent pipeline API instead:\n"
            "  result = brain.cv(k=5).predict(y=labels, algorithm='svm')\n"
            "The fluent API provides richer results (per-fold scores, predictions) "
            "and supports preprocessing chains (.normalize(), .reduce()).",
            DeprecationWarning,
            stacklevel=3,
        )

    # Resolve estimator
    estimator_obj = resolve_estimator(bd, estimator)

    # Resolve CV
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Validate y
    y = np.asarray(y)
    if y.shape[0] != bd.shape[0]:
        raise ValueError(
            f"y has {y.shape[0]} samples but data has {bd.shape[0]} samples"
        )

    # Get data as X for classification
    X_data = bd.data  # (n_samples, n_voxels)

    # Build pipeline with optional standardization
    if standardize:
        pipe = make_pipeline(StandardScaler(), clone(estimator_obj))
    else:
        pipe = clone(estimator_obj)

    # Dispatch by method
    if method == "whole_brain":
        # Use Pipeline infrastructure for whole-brain MVPA
        accuracy = mvpa_whole_brain_pipeline(bd, y, estimator, cv, groups, standardize)
    elif method == "searchlight":
        accuracy = mvpa_searchlight(
            bd, X_data, y, pipe, cv, groups, scoring, radius_mm, n_jobs, progress_bar
        )
    elif method == "roi":
        if roi_mask is None:
            raise ValueError("roi_mask required for method='roi'")
        accuracy = mvpa_roi(
            bd, X_data, y, pipe, cv, groups, scoring, roi_mask, n_jobs, progress_bar
        )

    # Wrap in BrainData
    result = bd[0].copy() if len(bd.shape) > 1 and bd.shape[0] > 1 else bd.copy()
    result.data = accuracy.reshape(1, -1) if accuracy.ndim == 1 else accuracy

    return result


def resolve_estimator(bd, estimator):
    """Resolve string shortcut to sklearn estimator.

    Args:
        bd: BrainData instance (unused, kept for API consistency).
        estimator: String shortcut or sklearn estimator object.

    Returns:
        Instantiated sklearn estimator.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import LinearSVC

    shortcuts = {
        "svm": lambda: LinearSVC(dual="auto", max_iter=10000),
        "logistic": lambda: LogisticRegression(max_iter=1000),
        "ridge": lambda: RidgeClassifier(),
        "lda": lambda: LinearDiscriminantAnalysis(),
    }

    if isinstance(estimator, str):
        if estimator not in shortcuts:
            raise ValueError(
                f"Unknown estimator: '{estimator}'. "
                f"Valid options: {list(shortcuts.keys())}"
            )
        return shortcuts[estimator]()

    # Validate sklearn API
    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise TypeError(
            f"estimator must have fit() and predict() methods. "
            f"Got: {type(estimator).__name__}"
        )

    return estimator


def mvpa_whole_brain(bd, X, y, pipe, cv, groups, scoring):
    """Whole-brain MVPA - single accuracy across all voxels.

    Legacy implementation using sklearn cross_val_score directly.
    Kept for searchlight/ROI methods that still use sklearn pipelines.

    Args:
        bd: BrainData instance (unused, kept for API consistency).
        X: Feature matrix, shape (n_samples, n_voxels).
        y: Labels, shape (n_samples,).
        pipe: sklearn pipeline or estimator.
        cv: Cross-validation splitter.
        groups: Group labels for CV.
        scoring: Scoring metric string.

    Returns:
        np.ndarray with single mean accuracy value.
    """
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring=scoring)
    return np.array([np.mean(scores)])


def mvpa_whole_brain_pipeline(bd, y, estimator, cv, groups, standardize):
    """Whole-brain MVPA using Pipeline infrastructure.

    Delegates to the fluent pipeline API for whole-brain classification,
    then extracts mean accuracy for backward compatibility.

    Args:
        bd: BrainData instance.
        y: Labels to predict.
        estimator: Estimator name ('svm', 'logistic', etc.).
        cv: Cross-validation splitter or int.
        groups: Group labels for CV.
        standardize: Whether to z-score features.

    Returns:
        np.ndarray with single mean accuracy value.
    """
    # Map estimator names to Pipeline algorithm names
    estimator_map = {
        "svm": "svm",
        "logistic": "logistic",
        "ridge": "ridge",
        "lda": "logistic",  # Approximate with logistic
    }

    # Get algorithm name (pass through if not in map)
    if isinstance(estimator, str):
        algorithm = estimator_map.get(estimator, estimator)
    else:
        # Custom estimator - fall back to legacy implementation
        from sklearn.base import clone
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        if standardize:
            pipe = make_pipeline(StandardScaler(), clone(estimator))
        else:
            pipe = clone(estimator)
        return mvpa_whole_brain(bd, bd.data, y, pipe, cv, groups, "accuracy")

    # Build CV parameters from cv argument
    if isinstance(cv, int):
        k = cv
    else:
        # For sklearn CV objects, extract n_splits
        k = getattr(cv, "n_splits", 5)

    # Build and execute pipeline
    pipeline = bd.cv(k=k, method="kfold", groups=groups)

    # Add normalization if requested
    if standardize:
        pipeline = pipeline.normalize(method="zscore")

    # Execute and get results
    cv_result = pipeline.predict(y=y, algorithm=algorithm)

    # Return mean accuracy as single-element array for backward compat
    return np.array([cv_result.mean_score])


def mvpa_searchlight(bd, X, y, pipe, cv, groups, scoring, radius_mm, n_jobs, progress_bar):
    """Searchlight MVPA - accuracy per voxel neighborhood.

    Args:
        bd: BrainData instance.
        X: Feature matrix, shape (n_samples, n_voxels).
        y: Labels, shape (n_samples,).
        pipe: sklearn pipeline or estimator.
        cv: Cross-validation splitter.
        groups: Group labels for CV.
        scoring: Scoring metric string.
        radius_mm: searchlight radius in mm.
        n_jobs: Number of parallel jobs.
        progress_bar: Whether to show progress bar.

    Returns:
        np.ndarray of accuracy values per voxel.
    """
    from joblib import Parallel, delayed
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    from .neighborhoods import compute_searchlight_neighborhoods

    # Get neighborhoods
    neighborhoods = compute_searchlight_neighborhoods(
        bd.mask, radius_mm=radius_mm, use_cache=True
    )

    def decode_sphere(center_idx, neighbor_indices):
        """Decode within a single sphere."""
        X_sphere = X[:, neighbor_indices]
        if X_sphere.shape[1] < 2:  # Skip tiny neighborhoods
            return np.nan
        try:
            scores = cross_val_score(
                clone(pipe), X_sphere, y, cv=cv, groups=groups, scoring=scoring
            )
            return np.mean(scores)
        except Exception:
            return np.nan

    # Collect all neighborhoods
    neighborhood_list = list(neighborhoods.iter_neighborhoods())

    # Progress bar setup
    if progress_bar:
        try:
            from tqdm import tqdm

            neighborhood_list = list(
                tqdm(
                    neighborhood_list,
                    desc="Searchlight",
                    total=neighborhoods.n_voxels,
                )
            )
        except ImportError:
            pass

    # Parallel execution
    if n_jobs == 1:
        accuracies = [decode_sphere(c, n) for c, n in neighborhood_list]
    else:
        accuracies = Parallel(n_jobs=n_jobs)(
            delayed(decode_sphere)(c, n) for c, n in neighborhood_list
        )

    return np.array(accuracies)


def mvpa_roi(bd, X, y, pipe, cv, groups, scoring, roi_mask, n_jobs, progress_bar):
    """ROI-based MVPA - accuracy per ROI, projected to voxel space.

    For each non-zero label in ``roi_mask``, uses all voxels within that ROI
    as features for a cross-validated decoder. Returns a voxel-shaped array
    where each voxel carries the accuracy of its containing ROI (voxels
    outside any ROI are NaN).

    Args:
        bd: BrainData instance.
        X: Feature matrix, shape (n_samples, n_voxels) — aligned to ``bd.mask``.
        y: Labels, shape (n_samples,).
        pipe: sklearn pipeline or estimator.
        cv: Cross-validation splitter.
        groups: Group labels for CV.
        scoring: Scoring metric string.
        roi_mask: Atlas/parcellation image or path. Resampled to ``bd.mask``
            space with nearest-neighbor interpolation if needed.
        n_jobs: Number of parallel jobs.
        progress_bar: Whether to show progress bar.

    Returns:
        np.ndarray of shape (n_voxels,) with per-voxel ROI accuracy.
    """
    from pathlib import Path

    import nibabel as nib
    from joblib import Parallel, delayed
    from nilearn.image import resample_to_img
    from nilearn.masking import apply_mask
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    if isinstance(roi_mask, (str, Path)):
        roi_mask = nib.load(roi_mask)

    # Align roi_mask to bd.mask (nearest-neighbor for integer labels)
    if roi_mask.shape != bd.mask.shape or not np.allclose(
        roi_mask.affine, bd.mask.affine
    ):
        roi_mask = resample_to_img(
            roi_mask,
            bd.mask,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )

    # Extract labels at the same voxels that bd.data indexes
    label_vec = apply_mask(roi_mask, bd.mask).astype(np.int64)  # (n_voxels,)
    unique_labels = np.unique(label_vec)
    unique_labels = unique_labels[unique_labels != 0]

    def decode_roi(roi_label):
        cols = label_vec == roi_label
        if not cols.any():
            return np.nan
        X_roi = X[:, cols]
        try:
            scores = cross_val_score(
                clone(pipe), X_roi, y, cv=cv, groups=groups, scoring=scoring
            )
            return float(np.mean(scores))
        except Exception:
            return np.nan

    iterator = unique_labels
    if progress_bar:
        try:
            from tqdm import tqdm

            iterator = tqdm(unique_labels, desc="ROI decoding")
        except ImportError:
            pass

    if n_jobs == 1:
        per_roi = [decode_roi(label) for label in iterator]
    else:
        per_roi = Parallel(n_jobs=n_jobs)(
            delayed(decode_roi)(label) for label in iterator
        )

    # Project ROI accuracies back to voxel space
    out = np.full(label_vec.shape, np.nan, dtype=float)
    for roi_label, acc in zip(unique_labels, per_roi):
        out[label_vec == roi_label] = acc
    return out
