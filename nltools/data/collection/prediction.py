"""
Prediction functions extracted from BrainCollection.

Contains predict, compute_contrasts, select_feature, and related helpers.
All BrainCollection methods converted to functions taking `bc` as first argument.
"""

from __future__ import annotations

import numpy as np

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nltools.data.collection import BrainCollection


def predict(
    bc,
    X: "np.ndarray | str | list | None" = None,
    y: "np.ndarray | None" = None,
    method: str = "whole_brain",
    estimator="svm",
    cv=5,
    groups: "np.ndarray | None" = None,
    roi_mask=None,
    radius: float = 10.0,
    scoring: str = "accuracy",
    standardize: bool = True,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> "BrainCollection":
    """
    Generate predictions for each subject in collection.

    This method supports two prediction modes determined by which parameter
    is provided:

    1. **Timeseries prediction** (X provided): Use fitted ridge model to
       predict voxel responses for new feature data.

    2. **MVPA decoding** (y provided): Train a classifier to predict labels
       from brain patterns using cross-validation.

    For MVPA, if this collection was created with by_run=True, you can
    use y=None to infer labels from _condition_labels and groups from
    _run_labels (leave-one-run-out CV).

    Args:
        bc: BrainCollection instance.
        X: Features for timeseries prediction. Can be:
            - np.ndarray: Shared features (same for all subjects)
            - str: Metadata column with per-subject feature paths
            - list: Per-subject feature arrays
        y: Labels for MVPA decoding. If None and _condition_labels exists,
            will use stored condition labels (from fit_glm with by_run=True).
        method: MVPA method - 'whole_brain', 'searchlight', or 'roi'.
        estimator: Classifier - 'svm', 'logistic', 'ridge', 'lda' or
            sklearn estimator instance.
        cv: Cross-validation strategy. If None and _run_labels exists,
            uses leave-one-group-out with run labels.
        groups: Group labels for GroupKFold/LeaveOneGroupOut. If None
            and _run_labels exists, uses stored run labels.
        roi_mask: Mask for ROI-based MVPA. Required if method='roi'.
        radius: Searchlight radius in mm (default 10.0).
        scoring: Scoring metric (default 'accuracy').
        standardize: If True, standardize features before classification.
        n_jobs: Parallel jobs for searchlight (-1 = all cores).
        show_progress: Show progress bar during fitting.

    Returns:
        BrainCollection with prediction results:
        - For timeseries: (n_timepoints, n_voxels) predicted responses
        - For MVPA: (1, n_voxels) accuracy values

    Examples:
        >>> # MVPA workflow with run-level betas
        >>> betas = bc.fit_glm(events=events, t_r=2.0, by_run=True)
        >>> accuracy = betas.predict(y=None, method='whole_brain')
        >>> # y and groups inferred from _condition_labels, _run_labels

        >>> # Explicit labels
        >>> accuracy = betas.predict(y=labels, method='searchlight')

        >>> # Timeseries prediction with ridge weights
        >>> weights = bc.fit_ridge(X=features, output='weights')
        >>> predictions = weights.predict(X=new_features)
    """
    import pandas as pd
    from nltools.data.collection import BrainCollection
    from nltools.data.collection.modeling import resolve_X, load_features
    from nltools.utils import attempt_to_import

    tqdm = attempt_to_import("tqdm", "tqdm")

    # Validate mutually exclusive modes
    if X is not None and y is not None:
        raise ValueError(
            "Cannot specify both X and y. Use X for timeseries prediction "
            "or y for MVPA decoding."
        )

    # Infer y from _condition_labels if available
    if y is None and X is None:
        if hasattr(bc, "_condition_labels") and bc._condition_labels:
            y = np.array(bc._condition_labels)
        else:
            raise ValueError(
                "Must provide X for timeseries prediction or y for MVPA. "
                "If using fit_glm(by_run=True), y can be inferred from "
                "_condition_labels."
            )

    # Infer groups from _run_labels if available
    if y is not None and groups is None:
        if hasattr(bc, "_run_labels") and bc._run_labels:
            groups = np.array(bc._run_labels)

    # Progress bar setup
    iterator = range(len(bc))
    if show_progress and tqdm is not None:
        desc = "Predicting (MVPA)" if y is not None else "Predicting (timeseries)"
        iterator = tqdm.tqdm(iterator, desc=desc, unit="subject")

    # Resolve per-subject features if X is provided
    X_list = None
    shared_X = None
    if X is not None:
        X_resolved = resolve_X(bc, X)
        if X_resolved is None:
            # Shared features
            shared_X = X
        else:
            X_list = X_resolved

    # Storage for results
    result_data_list = []
    result_metadata = []

    for i in iterator:
        # Load subject data
        bd = bc._load_item(i)
        metadata_row = bc._metadata.iloc[i]

        if X is not None:
            # Timeseries prediction mode
            if X_list is not None:
                subj_X = X_list[i]
                if isinstance(subj_X, (str, Path)):
                    subj_X = load_features(bc, subj_X)
            else:
                subj_X = shared_X

            result = bd.predict(X=subj_X)
        else:
            # MVPA mode
            result = bd.predict(
                y=y,
                method=method,
                estimator=estimator,
                cv=cv,
                groups=groups,
                roi_mask=roi_mask,
                radius=radius,
                scoring=scoring,
                standardize=standardize,
                n_jobs=n_jobs,
                show_progress=False,  # Disable per-subject progress
            )

        result_data_list.append(result)
        result_metadata.append(metadata_row.to_dict())

        # Unload to free memory
        bc.unload([i])

    # Build result collection
    result_collection = BrainCollection(
        result_data_list,
        mask=bc.mask,
        metadata=pd.DataFrame(result_metadata),
    )

    return result_collection


def compute_contrasts(
    bc,
    contrasts: "str | dict | np.ndarray | list",
) -> "BrainCollection | dict[str, BrainCollection]":
    """
    Compute contrasts from fitted GLM beta coefficients.

    Applies contrast weights to each subject's betas and returns a
    BrainCollection of contrast values suitable for group-level analysis.

    Must be called on a BrainCollection created by fit_glm() which has
    the _design_columns attribute set.

    Args:
        bc: BrainCollection instance.
        contrasts: Can be:
            - str: Contrast string using column names, e.g., "face - house"
            - dict: Multiple contrasts, e.g., {"main": "face - house", "avg": [0.5, 0.5]}
            - array/list: Numeric contrast vector, e.g., [1, -1]

    Returns:
        BrainCollection where each BrainData has shape (n_voxels,) containing
        the contrast values. If dict input, returns dict of BrainCollections.

    Raises:
        RuntimeError: If _design_columns not set (not from fit_glm)
        ValueError: If contrast vector length doesn't match number of regressors
        ValueError: If column name in string contrast not found

    Examples:
        >>> # Fit GLM and compute contrast
        >>> betas = bc.fit_glm(events=events_df, t_r=2.0)
        >>> contrast = betas.compute_contrasts("face - house")
        >>> # Group t-test on contrast
        >>> group_result = contrast.ttest()

        >>> # Multiple contrasts
        >>> contrasts = betas.compute_contrasts({
        ...     "face_vs_house": "face - house",
        ...     "face_vs_baseline": "face",
        ... })
        >>> face_vs_house_ttest = contrasts["face_vs_house"].ttest()
    """
    # Validate that this collection has design columns
    if not hasattr(bc, "_design_columns") or bc._design_columns is None:
        raise RuntimeError(
            "No design columns found. This method requires a BrainCollection "
            "created by fit_glm() which stores the task regressor names."
        )

    design_columns = bc._design_columns

    # Handle dict of contrasts
    if isinstance(contrasts, dict):
        results = {}
        for name, contrast_spec in contrasts.items():
            results[name] = compute_single_contrast(bc, contrast_spec, design_columns)
        return results

    # Single contrast
    return compute_single_contrast(bc, contrasts, design_columns)


def compute_single_contrast(
    bc,
    contrast: "str | np.ndarray | list",
    design_columns: list[str],
) -> "BrainCollection":
    """Compute a single contrast across all subjects.

    Args:
        bc: BrainCollection instance.
        contrast: Contrast specification (string, array, or list)
        design_columns: List of regressor names from fit_glm

    Returns:
        BrainCollection with contrast values for each subject
    """
    from nltools.data.collection import BrainCollection

    # Parse contrast to numeric vector
    if isinstance(contrast, str):
        contrast_vector = parse_contrast_string(bc, contrast, design_columns)
    else:
        contrast_vector = np.asarray(contrast)

    # Validate contrast vector length
    n_regressors = len(design_columns)
    if len(contrast_vector) != n_regressors:
        raise ValueError(
            f"Contrast vector length ({len(contrast_vector)}) must match "
            f"number of regressors ({n_regressors}). "
            f"Regressors: {design_columns}"
        )

    # Compute contrast for each subject
    contrast_data_list = []
    for i in range(len(bc)):
        bd = bc._load_item(i)

        # Compute weighted sum of betas
        # bd.data has shape (n_regressors, n_voxels)
        contrast_values = np.zeros(bd.shape[1])
        for j, weight in enumerate(contrast_vector):
            if weight != 0:
                contrast_values += weight * bd.data[j, :]

        # Create BrainData with contrast values
        contrast_bd = bd[0].copy()
        contrast_bd.data = contrast_values

        contrast_data_list.append(contrast_bd)

    # Build result collection
    return BrainCollection(
        contrast_data_list,
        mask=bc.mask,
        metadata=bc._metadata.copy(),
    )


def parse_contrast_string(
    bc,
    contrast_str: str,
    design_columns: list[str],
) -> np.ndarray:
    """Parse a contrast string into a numeric contrast vector.

    Args:
        bc: BrainCollection instance (unused, kept for API consistency).
        contrast_str: Contrast string like "A - B" or "2*A - B"
        design_columns: List of regressor column names

    Returns:
        Numeric contrast vector

    Raises:
        ValueError: If column name not found in design_columns
    """
    import re

    # Initialize contrast vector
    contrast_vector = np.zeros(len(design_columns))

    # Split by + and - (keeping the operators)
    tokens = re.split(r"(\+|\-)", contrast_str)
    tokens = [t.strip() for t in tokens if t.strip()]

    # Process tokens
    sign = 1  # Start with positive
    for token in tokens:
        if token == "+":
            sign = 1
        elif token == "-":
            sign = -1
        else:
            # Parse coefficient and variable
            if "*" in token:
                coef_str, var_name = token.split("*")
                coef = float(coef_str.strip())
                var_name = var_name.strip()
            else:
                coef = 1.0
                var_name = token

            # Find column index
            if var_name in design_columns:
                idx = design_columns.index(var_name)
                contrast_vector[idx] = sign * coef
            else:
                raise ValueError(
                    f"Column '{var_name}' not found in design columns. "
                    f"Available: {design_columns}"
                )

    return contrast_vector


def select_feature(
    bc,
    feature: "int | str",
) -> "BrainCollection":
    """
    Select a single feature's weights across all subjects.

    Used after fit_ridge() to extract weights for a specific feature
    for group-level analysis (e.g., t-test on feature weights).

    Must be called on a BrainCollection created by fit_ridge() where
    each subject has shape (n_features, n_voxels).

    Args:
        bc: BrainCollection instance.
        feature: Feature to select. Can be:
            - int: Feature index (0-based)
            - str: Feature name (requires _feature_names attribute)

    Returns:
        BrainCollection where each BrainData has shape (n_voxels,)
        containing the weights for the specified feature.

    Raises:
        IndexError: If feature index out of range
        KeyError: If feature name not found in _feature_names
        RuntimeError: If string feature given but _feature_names not set

    Examples:
        >>> # Fit ridge and select feature
        >>> weights = bc.fit_ridge(X=features, alpha=1.0)
        >>> feature_0 = weights.select_feature(0)
        >>> # Group t-test on first feature's weights
        >>> group_result = feature_0.ttest()

        >>> # By name (if features had column names)
        >>> face_weights = weights.select_feature("face_response")
    """
    from nltools.data.collection import BrainCollection

    # Resolve feature name to index
    if isinstance(feature, str):
        if not hasattr(bc, "_feature_names") or bc._feature_names is None:
            raise RuntimeError(
                "Cannot select feature by name: _feature_names not set. "
                "Use integer index or pass features with column names to fit_ridge()."
            )
        if feature not in bc._feature_names:
            raise KeyError(
                f"Feature '{feature}' not found. Available: {bc._feature_names}"
            )
        feature_idx = bc._feature_names.index(feature)
    else:
        feature_idx = feature

    # Extract feature weights for each subject
    feature_data_list = []
    for i in range(len(bc)):
        bd = bc._load_item(i)

        # Validate index
        if feature_idx < 0 or feature_idx >= bd.shape[0]:
            raise IndexError(
                f"Feature index {feature_idx} out of range for subject {i} "
                f"with {bd.shape[0]} features."
            )

        # Extract single feature's weights
        feature_values = bd.data[feature_idx, :]

        # Create BrainData with feature weights
        feature_bd = bd[0].copy()
        feature_bd.data = feature_values

        feature_data_list.append(feature_bd)

    # Build result collection
    return BrainCollection(
        feature_data_list,
        mask=bc.mask,
        metadata=bc._metadata.copy(),
    )
