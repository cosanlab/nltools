"""
Modeling functions extracted from BrainCollection.

Contains GLM fitting, Ridge fitting, design matrix building, and related helpers.
All BrainCollection methods converted to functions taking `bc` as first argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nltools.data.braindata import BrainData
    from nltools.data.collection import BrainCollection
    from nltools.data.collection.pipeline import (
        BrainCollectionPipeline,
        FittedBrainCollection,
    )


def _build_subject_design_matrix(
    events: pd.DataFrame,
    n_scans: int,
    t_r: float,
    confounds: pd.DataFrame | Path | str | None = None,
    confound_columns: list[str] | None = None,
    hrf_model: str = "spm",
    drift_model: str = "cosine",
    high_pass: float = 0.01,
) -> tuple[pd.DataFrame, list[str]]:
    """Build complete design matrix for a subject.

    Combines task design (from events) with subject-specific confounds.

    Args:
        events: Task events DataFrame with 'onset', 'duration', 'trial_type' columns
        n_scans: Number of scans/timepoints
        t_r: Repetition time in seconds
        confounds: Subject confounds - DataFrame, path to TSV, or None
        confound_columns: Which confound columns to include (None = all)
        hrf_model: HRF model for task regressors ('spm', 'glover', etc.)
        drift_model: Drift model ('cosine', 'polynomial', None)
        high_pass: High-pass filter cutoff in Hz

    Returns:
        Tuple of:
            - design_matrix: Complete design matrix (task + confounds + drift)
            - task_columns: List of task regressor column names

    Example:
        >>> events = pd.DataFrame({
        ...     'onset': [0, 10, 20],
        ...     'duration': [2, 2, 2],
        ...     'trial_type': ['face', 'house', 'face']
        ... })
        >>> dm, task_cols = _build_subject_design_matrix(events, 100, 2.0)
        >>> print(task_cols)
        ['face', 'house']
    """
    from nilearn.glm.first_level import make_first_level_design_matrix

    # Create frame times
    frame_times = np.arange(n_scans) * t_r

    # Build task design matrix
    task_dm = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
    )

    # Identify task columns (everything except drift and constant)
    drift_cols = [
        c for c in task_dm.columns if c.startswith("drift_") or c == "constant"
    ]
    task_columns = [c for c in task_dm.columns if c not in drift_cols]

    # Load confounds if provided
    if confounds is not None:
        if isinstance(confounds, (str, Path)):
            confounds_path = Path(confounds)
            if not confounds_path.exists():
                raise FileNotFoundError(f"Confounds file not found: {confounds_path}")
            # Detect separator (TSV vs CSV)
            sep = "\t" if confounds_path.suffix in [".tsv", ".txt"] else ","
            confounds_df = pd.read_csv(confounds_path, sep=sep)
        else:
            confounds_df = confounds

        # Select columns if specified
        if confound_columns is not None:
            missing = set(confound_columns) - set(confounds_df.columns)
            if missing:
                raise ValueError(
                    f"Confound columns not found: {missing}. "
                    f"Available: {list(confounds_df.columns)}"
                )
            confounds_df = confounds_df[confound_columns]

        # Validate length
        if len(confounds_df) != n_scans:
            raise ValueError(
                f"Confounds have {len(confounds_df)} rows but data has {n_scans} scans. "
                "Lengths must match."
            )

        # Handle NaN values in confounds (common for first few rows of derivatives)
        confounds_df = confounds_df.fillna(0)

        # Align confounds index with design matrix frame_times index
        confounds_df.index = task_dm.index

        # Concatenate: task_dm already has drift and constant, add confounds before them
        # Reorder: task | confounds | drift | constant
        full_dm = pd.concat(
            [task_dm[task_columns], confounds_df, task_dm[drift_cols]],
            axis=1,
        )
    else:
        full_dm = task_dm

    return full_dm, task_columns


def _fit_glm_by_run(
    bd: "BrainData",
    events: pd.DataFrame,
    runs: list,
    run_column: str,
    run_lengths: int | list[int] | None,
    t_r: float,
    confounds: pd.DataFrame | None,
    confound_columns: list[str] | None,
    hrf_model: str,
    drift_model: str,
    high_pass: float,
    scale: bool,
    scale_value: float,
) -> tuple["BrainData", list[str], list[str], list]:
    """Fit GLM separately for each run and stack betas.

    Args:
        bd: Subject's BrainData (concatenated timeseries)
        events: Events DataFrame with run column
        runs: Unique run identifiers (sorted)
        run_column: Column name for run identifier
        run_lengths: TRs per run (int for uniform, list for variable, None to infer)
        t_r: Repetition time
        confounds: Subject confounds (concatenated) or None
        confound_columns: Columns to extract from confounds
        hrf_model: HRF model for design matrix
        drift_model: Drift model
        high_pass: High-pass filter cutoff
        scale: Whether to apply percent signal change scaling
        scale_value: Scaling value

    Returns:
        Tuple of:
            - BrainData with stacked run-level betas (n_runs * n_conditions, n_voxels)
            - task_columns: List of condition names
            - condition_labels: Condition label for each beta row
            - run_labels: Run label for each beta row
    """
    n_scans = bd.shape[0]
    n_runs = len(runs)

    # Resolve run lengths
    if run_lengths is None:
        # Try to infer equal-length runs
        if n_scans % n_runs != 0:
            raise ValueError(
                f"Cannot infer run lengths: {n_scans} scans not evenly divisible by "
                f"{n_runs} runs. Please provide run_lengths parameter."
            )
        run_length_list = [n_scans // n_runs] * n_runs
    elif isinstance(run_lengths, int):
        run_length_list = [run_lengths] * n_runs
    else:
        run_length_list = list(run_lengths)

    # Validate total matches
    if sum(run_length_list) != n_scans:
        raise ValueError(
            f"run_lengths sum ({sum(run_length_list)}) does not match "
            f"total scans ({n_scans})"
        )

    # Calculate run start indices
    run_starts = [0]
    for length in run_length_list[:-1]:
        run_starts.append(run_starts[-1] + length)

    # Storage for run-level results
    all_betas = []
    condition_labels = []
    run_labels = []
    task_columns = None

    for run_idx, run_id in enumerate(runs):
        # Get run boundaries
        start_tr = run_starts[run_idx]
        end_tr = start_tr + run_length_list[run_idx]
        n_run_scans = run_length_list[run_idx]

        # Slice data for this run
        run_bd = bd[start_tr:end_tr]

        # Filter events to this run and adjust onsets (assuming run-relative onsets)
        run_events = events[events[run_column] == run_id].copy()
        # Remove run column for design matrix building
        run_events = run_events.drop(columns=[run_column])

        # Slice confounds if provided
        run_confounds = None
        if confounds is not None:
            run_confounds = confounds.iloc[start_tr:end_tr].reset_index(drop=True)

        # Build design matrix for this run
        dm, task_cols = _build_subject_design_matrix(
            events=run_events,
            n_scans=n_run_scans,
            t_r=t_r,
            confounds=run_confounds,
            confound_columns=confound_columns,
            hrf_model=hrf_model,
            drift_model=drift_model,
            high_pass=high_pass,
        )

        # Store task columns from first run
        if task_columns is None:
            task_columns = task_cols

        # Apply scaling if requested
        if scale:
            run_bd = run_bd.scale(scale_value)

        # Fit GLM
        run_bd.fit(model="glm", X=dm)

        # Extract task betas only
        task_indices = [dm.columns.get_loc(col) for col in task_cols]
        run_betas = run_bd.glm_betas.data[task_indices, :]

        # Store betas and labels
        all_betas.append(run_betas)
        condition_labels.extend(task_cols)
        run_labels.extend([run_id] * len(task_cols))

    # Stack all run betas
    stacked_betas = np.vstack(all_betas)

    # Create output BrainData
    result = bd[0].copy()
    result.data = stacked_betas
    result._design_columns = task_columns

    assert task_columns is not None  # set on first run iteration
    return result, task_columns, condition_labels, run_labels


def cv(
    bc,
    k: int | None = None,
    scheme: str = "kfold",
    split_by: str | None = None,
    groups: np.ndarray | None = None,
    random_state: int | None = None,
    **kwargs,
) -> "BrainCollectionPipeline":
    """Create a cross-validation pipeline for multi-subject analysis.

    Returns a pipeline object that enables fluent, chainable transforms
    with cross-validation across subjects or runs.

    Args:
        bc: BrainCollection instance.
        k: Number of folds (for kfold scheme). Defaults to 5.
        scheme: CV scheme type. Options:
            - 'kfold': k-fold cross-validation on pooled data
            - 'loso': leave-one-subject-out (one image held out per fold)
            - 'loro': leave-one-run-out (requires groups)
        split_by: Metadata column for group splits.
            If provided and groups is None, gets groups from bc.metadata[split_by].
        groups: Explicit group labels for CV splits.
        random_state: Random seed for reproducibility.
        **kwargs: Additional arguments passed to CVScheme.

    Returns:
        BrainCollectionPipeline: Pipeline for method chaining.

    Examples:
        >>> # Leave-one-subject-out classification
        >>> result = bc.cv(scheme='loso').normalize().predict(subject_labels, algorithm='svm')
        >>> print(f"Mean accuracy: {result.mean_score:.2%}")

        >>> # With preprocessing
        >>> result = (bc
        ...     .cv(scheme='loso')
        ...     .normalize()
        ...     .reduce(n_components=50)
        ...     .predict(labels))

        >>> # Run-based CV with metadata
        >>> result = bc.cv(scheme='loro', split_by='run').predict(y)

    See Also:
        BrainCollectionPipeline: For available transforms and terminals.
        CVScheme: For CV scheme configuration details.
    """
    from nltools.pipelines.cv import CVScheme
    from nltools.data.collection.pipeline import BrainCollectionPipeline

    # Handle split_by -> groups conversion from metadata
    if groups is None and split_by is not None:
        if bc.metadata is not None and split_by in bc.metadata.columns:
            groups = np.array(bc.metadata[split_by])

    # Create CV scheme
    cv_scheme = CVScheme(
        k=k,
        scheme=scheme,
        split_by=split_by,
        random_state=random_state,
        **kwargs,
    )

    return BrainCollectionPipeline(bc, cv=cv_scheme, groups=groups)


def fit(
    bc,
    model: str,
    X: "pd.DataFrame | np.ndarray | str | list",
    cv: int | None = None,
    scale: bool = True,
    scale_value: float = 100.0,
    show_progress: bool = True,
    **kwargs,
) -> "FittedBrainCollection":
    """
    Fit a model to each subject in the collection.

    Unified fitting method that shadows BrainData.fit() API for multi-subject
    analysis. Dispatches to model-specific implementations based on the
    model parameter.

    Args:
        bc: BrainCollection instance.
        model: Model type - 'glm' or 'ridge'
        X: Design/feature matrix. Can be:
            - pd.DataFrame/DesignMatrix: Shared (used for all subjects)
            - np.ndarray: Shared array (used for all subjects)
            - str: Column name in metadata pointing to file paths
            - list: Per-subject list of DataFrames/arrays/paths
        cv: Cross-validation folds (Ridge only). Default is None for GLM,
            5 for Ridge when output='scores'.
        scale: If True, apply percent signal change scaling before fitting.
        scale_value: Scaling value (default 100.0 for percent signal change).
        show_progress: Show progress bar during fitting.
        **kwargs: Model-specific arguments passed to _fit_glm or _fit_ridge:
            - GLM: return_stats, save
            - Ridge: alpha, output, save, backend, random_state

    Returns:
        FittedBrainCollection wrapping the fitted results. Supports:

        - ``.results``: Access underlying BrainCollection(s) directly
        - ``.betas``: Convenience accessor for beta coefficients (GLM)
        - ``.pool()``: Aggregate across subjects for group analysis

        The underlying results contain:

        - GLM: Beta coefficients (n_regressors, n_voxels) per subject
        - Ridge: Scores or weights depending on 'output' kwarg

        If return_stats (GLM) or output='both' (Ridge), results is a dict.

    Examples:
        >>> # GLM with shared design matrix
        >>> fitted = bc.fit(model='glm', X=dm)
        >>> betas = fitted.results  # Access BrainCollection directly
        >>>
        >>> # Two-stage analysis with pool()
        >>> pool = bc.fit(model='glm', X=dm).pool(param='beta')
        >>> t_map = pool.fit(model='ttest', contrast='A-B')
        >>>
        >>> # GLM with per-subject design matrices
        >>> fitted = bc.fit(model='glm', X=[dm1, dm2, dm3])
        >>>
        >>> # Ridge encoding model with CV scores
        >>> fitted = bc.fit(model='ridge', X=features, cv=5)
        >>> scores = fitted.results

    See Also:
        fit_from_events: Convenience method for event-based GLM workflows
        fit_glm: Legacy GLM fitting (use fit_from_events instead)
        fit_ridge: Legacy Ridge fitting (use fit(..., model='ridge') instead)
    """
    from nltools.data.collection.pipeline import FittedBrainCollection

    if model == "glm":
        results = fit_glm_internal(
            bc,
            X=X,
            scale=scale,
            scale_value=scale_value,
            show_progress=show_progress,
            **kwargs,
        )
        # Extract condition names from results
        condition_names = None
        if isinstance(results, dict):
            betas = results.get("betas")
            if betas is not None and hasattr(betas, "_design_columns"):
                condition_names = betas._design_columns
        elif hasattr(results, "_design_columns"):
            condition_names = results._design_columns

        return FittedBrainCollection(
            brain_collection=bc,
            fitted_results=results,
            model=model,
            condition_names=condition_names,
        )
    elif model == "ridge":
        # Handle cv default for Ridge
        if cv is None:
            output = kwargs.get("output", "scores")
            if output in ("scores", "both"):
                cv = 5  # Default for scores
        results = fit_ridge(
            bc,
            X=X,
            cv=cv,
            scale=scale,
            scale_value=scale_value,
            show_progress=show_progress,
            **kwargs,
        )
        return FittedBrainCollection(
            brain_collection=bc,
            fitted_results=results,
            model=model,
            condition_names=None,  # Ridge doesn't have condition names
        )
    else:
        raise ValueError(f"Unknown model: '{model}'. Supported: 'glm', 'ridge'")


def fit_glm(
    bc,
    events: pd.DataFrame,
    t_r: float,
    confounds: str | list[pd.DataFrame | Path | str] | None = None,
    confound_columns: list[str] | None = None,
    hrf_model: str = "spm",
    drift_model: str = "cosine",
    high_pass: float = 0.01,
    scale: bool = True,
    scale_value: float = 100.0,
    return_stats: list[str] | None = None,
    return_residuals: bool = False,
    save: dict[str, str] | None = None,
    show_progress: bool = True,
    by_run: bool = False,
    run_column: str = "run",
    run_lengths: int | list[int] | None = None,
) -> "BrainCollection | dict[str, BrainCollection]":
    """
    Fit GLM to each subject in collection.

    Memory-efficient first-level GLM analysis that processes subjects
    one at a time. Returns a BrainCollection of beta coefficients for
    task regressors (confounds and drift terms are fit but not returned).

    Args:
        bc: BrainCollection instance.
        events: Task events DataFrame with onset, duration, trial_type columns.
            This is shared across all subjects (same experimental paradigm).
            If by_run=True, must also have a run column.
        t_r: Repetition time (TR) in seconds.
        confounds: Subject-specific confounds. Can be:
            - str: Column name in metadata pointing to confound file paths
            - list: List of DataFrames or paths, one per subject
            - None: No confounds (only task + drift terms)
        confound_columns: Columns to extract from confound files.
            If None and confounds provided, uses all columns.
        hrf_model: HRF model for convolution ('spm', 'glover', 'fir', etc.)
        drift_model: Drift model ('cosine', 'polynomial', None)
        high_pass: High-pass filter cutoff in Hz (default 0.01)
        scale: If True, apply percent signal change scaling before fitting.
        scale_value: Scaling value (default 100.0 for percent signal change).
        return_stats: Optional list of statistics to return as separate
            BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'.
        return_residuals: If True, return residuals (same as return_stats=['residual']).
        save: Dict mapping output type to path template, e.g.
            ``{'betas': 'output/{subject}_betas.nii.gz',
            't': 'output/{subject}_tstat.nii.gz'}``.
            Supports {subject}, {session}, {idx}, and other metadata columns.
        show_progress: Show progress bar during fitting.
        by_run: If True, fit GLM separately per run and return run-level betas.
            This enables MVPA decoding with leave-one-run-out CV.
            Each subject will have (n_runs * n_conditions, n_voxels) betas.
        run_column: Column name in events identifying runs (default 'run').
        run_lengths: Number of TRs per run. Required when by_run=True.

            - int: All runs have same length
            - list of int: Different length per run
            - None: Will attempt to infer equal-length runs from total scans

    Returns:
        BrainCollection where each BrainData has shape:

        - (n_task_regressors, n_voxels) if by_run=False (default)
        - (n_runs * n_task_regressors, n_voxels) if by_run=True

        The ``._design_columns`` attribute stores task regressor names.
        If by_run=True, also stores ``._condition_labels`` and ``._run_labels``.
        If return_stats specified, returns dict with keys 'betas', 't', etc.

    Examples:
        >>> # Basic GLM fit
        >>> betas = bc.fit_glm(events=events_df, t_r=2.0)
        >>> # Group t-test on first regressor
        >>> group_t = betas[:, 0].ttest()

        >>> # Run-level betas for MVPA decoding
        >>> betas = bc.fit_glm(events=events_df, t_r=2.0, by_run=True)
        >>> # betas._condition_labels = ['face', 'house', 'face', 'house', ...]
        >>> # betas._run_labels = [1, 1, 2, 2, 3, 3, ...]
        >>> accuracy = betas.predict(y=None, method='searchlight')

        >>> # With confounds from metadata column
        >>> betas = bc.fit_glm(
        ...     events=events_df,
        ...     t_r=2.0,
        ...     confounds='confound_file',  # column name in metadata
        ...     confound_columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        ... )
    """
    from nltools.data.collection import BrainCollection
    from nltools.utils import attempt_to_import

    tqdm = attempt_to_import("tqdm", "tqdm")

    # Handle return_residuals shorthand
    if return_residuals and return_stats is None:
        return_stats = ["residual"]
    elif return_residuals and "residual" not in return_stats:
        return_stats = list(return_stats) + ["residual"]

    # Validate return_stats
    valid_stats = {"t", "r2", "p", "se", "residual"}
    if return_stats is not None:
        invalid = set(return_stats) - valid_stats
        if invalid:
            raise ValueError(
                f"Invalid return_stats: {invalid}. Valid options: {valid_stats}"
            )

    # Validate by_run parameters
    if by_run:
        if run_column not in events.columns:
            raise ValueError(
                f"by_run=True requires '{run_column}' column in events. "
                f"Available columns: {list(events.columns)}"
            )
        runs = sorted(events[run_column].unique())
        if return_stats is not None:
            raise NotImplementedError(
                "return_stats is not yet supported with by_run=True. "
                "Only beta coefficients are returned."
            )
    else:
        runs = None

    # Resolve confounds to per-subject list
    confounds_list = resolve_confounds(bc, confounds)

    # Progress bar setup
    iterator = range(len(bc))
    if show_progress and tqdm is not None:
        iterator = tqdm.tqdm(iterator, desc="Fitting GLM", unit="subject")

    # Storage for results
    beta_data_list = []
    beta_metadata = []
    stat_data = {stat: [] for stat in (return_stats or [])}
    task_columns = None  # Will be set on first subject
    # For by_run mode: store labels (same for all subjects)
    all_condition_labels = None
    all_run_labels = None

    for i in iterator:
        # Load subject data
        bd = bc._load_item(i)
        metadata_row = bc._metadata.iloc[i]
        n_scans = bd.shape[0]

        # Get subject-specific confounds
        subj_confounds = confounds_list[i] if confounds_list else None

        if by_run:
            # ===== BY_RUN MODE: Fit GLM separately per run =====
            # Load confounds as DataFrame for slicing
            if subj_confounds is not None:
                if isinstance(subj_confounds, (str, Path)):
                    conf_path = Path(subj_confounds)
                    sep = "\t" if conf_path.suffix in [".tsv", ".txt"] else ","
                    subj_confounds = pd.read_csv(conf_path, sep=sep)
                    if confound_columns:
                        subj_confounds = subj_confounds[confound_columns]
                    subj_confounds = subj_confounds.fillna(0)

            task_betas, task_cols, cond_labels, run_lbls = _fit_glm_by_run(
                bd=bd,
                events=events,
                runs=runs,
                run_column=run_column,
                run_lengths=run_lengths,
                t_r=t_r,
                confounds=subj_confounds,
                confound_columns=confound_columns,
                hrf_model=hrf_model,
                drift_model=drift_model,
                high_pass=high_pass,
                scale=scale,
                scale_value=scale_value,
            )

            # Store task columns and labels from first subject
            if task_columns is None:
                task_columns = task_cols
                all_condition_labels = cond_labels
                all_run_labels = run_lbls

        else:
            # ===== STANDARD MODE: Single GLM per subject =====
            # Build design matrix
            dm, task_cols = _build_subject_design_matrix(
                events=events,
                n_scans=n_scans,
                t_r=t_r,
                confounds=subj_confounds,
                confound_columns=confound_columns,
                hrf_model=hrf_model,
                drift_model=drift_model,
                high_pass=high_pass,
            )

            # Store task columns for later
            if task_columns is None:
                task_columns = task_cols

            # Apply scaling if requested
            if scale:
                bd = bd.scale(scale_value)

            # Fit GLM
            bd.fit(model="glm", X=dm)

            # Extract task betas only (not confounds/drift)
            task_indices = [dm.columns.get_loc(col) for col in task_cols]
            task_betas_data = bd.glm_betas.data[task_indices, :]

            # Create BrainData for task betas by copying structure
            task_betas = bd[0].copy()
            task_betas.data = task_betas_data
            task_betas._design_columns = task_cols  # Store for contrast parsing

        # Save if requested
        if save and "betas" in save:
            from nltools.data.collection import _resolve_save_path

            save_path = _resolve_save_path(save["betas"], metadata_row, i)
            task_betas.write(str(save_path))

        beta_data_list.append(task_betas)
        beta_metadata.append(metadata_row.to_dict())

        # Extract optional stats (standard mode only, validated earlier)
        if return_stats:
            for stat in return_stats:
                if stat == "t":
                    stat_data_arr = bd.glm_t.data[task_indices, :]
                elif stat == "p":
                    stat_data_arr = bd.glm_p.data[task_indices, :]
                elif stat == "se":
                    stat_data_arr = bd.glm_se.data[task_indices, :]
                elif stat == "r2":
                    stat_data_arr = bd.glm_r2.data  # Shape (1, n_voxels)
                elif stat == "residual":
                    stat_data_arr = bd.glm_residual.data  # Shape (n_scans, n_voxels)

                # Create BrainData by copying structure
                stat_bd = bd[0].copy()
                stat_bd.data = stat_data_arr

                if save and stat in save:
                    from nltools.data.collection import _resolve_save_path

                    save_path = _resolve_save_path(save[stat], metadata_row, i)
                    stat_bd.write(str(save_path))

                stat_data[stat].append(stat_bd)

        # Unload to free memory (only works for path-based collections)
        bc.unload([i])

    # Build result collection
    beta_collection = BrainCollection(
        beta_data_list,
        mask=bc.mask,
        metadata=pd.DataFrame(beta_metadata),
    )
    beta_collection._design_columns = task_columns

    # Store run-level labels for MVPA workflows
    if by_run:
        beta_collection._condition_labels = all_condition_labels
        beta_collection._run_labels = all_run_labels

    # Return based on what was requested
    if return_stats:
        result = {"betas": beta_collection}
        for stat in return_stats:
            stat_collection = BrainCollection(
                stat_data[stat],
                mask=bc.mask,
                metadata=pd.DataFrame(beta_metadata),
            )
            result[stat] = stat_collection
        return result

    return beta_collection


def fit_from_events(
    bc,
    events: pd.DataFrame,
    t_r: float,
    confounds: str | list[pd.DataFrame | Path | str] | None = None,
    confound_columns: list[str] | None = None,
    hrf_model: str = "spm",
    drift_model: str = "cosine",
    high_pass: float = 0.01,
    scale: bool = True,
    scale_value: float = 100.0,
    return_stats: list[str] | None = None,
    return_residuals: bool = False,
    save: dict[str, str] | None = None,
    show_progress: bool = True,
    by_run: bool = False,
    run_column: str = "run",
    run_lengths: int | list[int] | None = None,
) -> "BrainCollection | dict[str, BrainCollection]":
    """
    Build design matrices from events and fit GLM to each subject.

    Convenience method for event-based experimental designs. Builds
    nilearn-compatible design matrices from the events DataFrame and
    fits a GLM to each subject in the collection.

    This is the recommended method for typical task-based fMRI analysis
    where you have event timing information. For more control, use
    fit(model='glm', X=design_matrices) with pre-built design matrices.

    Args:
        bc: BrainCollection instance.
        events: Task events DataFrame with onset, duration, trial_type columns.
            This is shared across all subjects (same experimental paradigm).
            If by_run=True, must also have a run column.
        t_r: Repetition time (TR) in seconds.
        confounds: Subject-specific confounds. Can be:
            - str: Column name in metadata pointing to confound file paths
            - list: List of DataFrames or paths, one per subject
            - None: No confounds (only task + drift terms)
        confound_columns: Columns to extract from confound files.
            If None and confounds provided, uses all columns.
        hrf_model: HRF model for convolution ('spm', 'glover', 'fir', etc.)
        drift_model: Drift model ('cosine', 'polynomial', None)
        high_pass: High-pass filter cutoff in Hz (default 0.01)
        scale: If True, apply percent signal change scaling before fitting.
        scale_value: Scaling value (default 100.0 for percent signal change).
        return_stats: Optional list of statistics to return as separate
            BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'.
        return_residuals: If True, return residuals (same as return_stats=['residual']).
        save: Dict mapping output type to path template.
        show_progress: Show progress bar during fitting.
        by_run: If True, fit GLM separately per run and return run-level betas.
            This enables MVPA decoding with leave-one-run-out CV.
        run_column: Column name in events identifying runs (default 'run').
        run_lengths: Number of TRs per run. Required when by_run=True.

    Returns:
        BrainCollection of beta coefficients for task regressors.
        If return_stats specified, returns dict with keys 'betas', 't', etc.

    Examples:
        >>> # Basic GLM fit from events
        >>> betas = bc.fit_from_events(events=events_df, t_r=2.0)
        >>> group_t = betas.ttest()
        >>>
        >>> # With confounds from metadata column
        >>> betas = bc.fit_from_events(
        ...     events=events_df,
        ...     t_r=2.0,
        ...     confounds='confound_file',
        ...     confound_columns=['trans_x', 'trans_y', 'trans_z']
        ... )
        >>>
        >>> # Run-level betas for MVPA
        >>> betas = bc.fit_from_events(events=events_df, t_r=2.0, by_run=True)

    See Also:
        fit: Unified fit method that accepts pre-built design matrices
        _fit_glm: Internal method for design matrix-based fitting
    """
    return fit_glm(
        bc,
        events=events,
        t_r=t_r,
        confounds=confounds,
        confound_columns=confound_columns,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
        scale=scale,
        scale_value=scale_value,
        return_stats=return_stats,
        return_residuals=return_residuals,
        save=save,
        show_progress=show_progress,
        by_run=by_run,
        run_column=run_column,
        run_lengths=run_lengths,
    )


def fit_glm_internal(
    bc,
    X: "pd.DataFrame | np.ndarray | str | list",
    scale: bool = True,
    scale_value: float = 100.0,
    return_stats: list[str] | None = None,
    save: dict[str, str] | None = None,
    show_progress: bool = True,
) -> "BrainCollection | dict[str, BrainCollection]":
    """Internal GLM fitting with design matrix input.

    Core implementation that accepts DesignMatrix/DataFrame directly.
    Called by fit(model='glm') and fit_from_events().

    Args:
        bc: BrainCollection instance.
        X: Design matrix. Can be:
            - pd.DataFrame/DesignMatrix: Shared (used for all subjects)
            - np.ndarray: Shared array (converted to DataFrame internally)
            - str: Column name in metadata pointing to file paths
            - list: Per-subject list of DataFrames/arrays/paths
        scale: If True, apply percent signal change scaling before fitting.
        scale_value: Scaling value (default 100.0 for percent signal change).
        return_stats: Optional list of statistics to return as separate
            BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'.
        save: Dict mapping output type to path template.
        show_progress: Show progress bar during fitting.

    Returns:
        BrainCollection of betas, or dict with betas + requested stats.
    """
    from nltools.data.collection import BrainCollection
    from nltools.utils import attempt_to_import

    tqdm = attempt_to_import("tqdm", "tqdm")

    # Validate return_stats
    valid_stats = {"t", "r2", "p", "se", "residual"}
    if return_stats is not None:
        invalid = set(return_stats) - valid_stats
        if invalid:
            raise ValueError(
                f"Invalid return_stats: {invalid}. Valid options: {valid_stats}"
            )

    # Resolve X to per-subject list (or None if shared)
    X_list = resolve_X(bc, X)

    # Progress bar setup
    iterator = range(len(bc))
    if show_progress and tqdm is not None:
        iterator = tqdm.tqdm(iterator, desc="Fitting GLM", unit="subject")

    # Storage for results
    beta_data_list = []
    beta_metadata = []
    stat_data = {stat: [] for stat in (return_stats or [])}
    design_columns = None  # Will be set on first subject

    for i in iterator:
        # Load subject data
        bd = bc._load_item(i)
        metadata_row = bc._metadata.iloc[i]

        # Get subject-specific design matrix
        X_subj = X_list[i] if X_list else X

        # Load from file if needed
        if isinstance(X_subj, (str, Path)):
            X_subj = load_design_matrix(bc, X_subj)

        # Convert array to DataFrame if needed
        if isinstance(X_subj, np.ndarray):
            X_subj = pd.DataFrame(
                X_subj, columns=[f"col_{j}" for j in range(X_subj.shape[1])]
            )

        # Store design columns for result metadata
        if design_columns is None:
            design_columns = list(X_subj.columns)

        # Validate shapes match
        if X_subj.shape[0] != bd.shape[0]:
            raise ValueError(
                f"Subject {i}: X has {X_subj.shape[0]} samples but data has "
                f"{bd.shape[0]} samples. Shapes must match."
            )

        # Apply scaling if requested (scale=False since we scale here)
        if scale:
            bd = bd.scale(scale_value)

        # Fit GLM using BrainData.fit()
        bd.fit(model="glm", X=X_subj, scale=False)

        # Extract betas
        betas = bd[0].copy()
        betas.data = bd.glm_betas.data
        betas._design_columns = design_columns

        # Save if requested
        if save and "betas" in save:
            from nltools.data.collection import _resolve_save_path

            save_path = _resolve_save_path(save["betas"], metadata_row, i)
            betas.write(str(save_path))

        beta_data_list.append(betas)
        beta_metadata.append(metadata_row.to_dict())

        # Extract optional stats
        if return_stats:
            for stat in return_stats:
                if stat == "t":
                    stat_bd = bd[0].copy()
                    stat_bd.data = bd.glm_t.data
                elif stat == "p":
                    stat_bd = bd[0].copy()
                    stat_bd.data = bd.glm_p.data
                elif stat == "se":
                    stat_bd = bd[0].copy()
                    stat_bd.data = bd.glm_se.data
                elif stat == "r2":
                    stat_bd = bd[0].copy()
                    stat_bd.data = bd.glm_r2.data
                elif stat == "residual":
                    stat_bd = bd[0].copy()
                    stat_bd.data = bd.glm_residual.data

                stat_bd._design_columns = design_columns

                if save and stat in save:
                    from nltools.data.collection import _resolve_save_path

                    save_path = _resolve_save_path(save[stat], metadata_row, i)
                    stat_bd.write(str(save_path))

                stat_data[stat].append(stat_bd)

    # Build result collection
    result_metadata = pd.DataFrame(beta_metadata)
    beta_collection = BrainCollection(
        beta_data_list, mask=bc._mask, metadata=result_metadata, lazy=False
    )
    beta_collection._design_columns = design_columns

    # Return stats if requested
    if return_stats:
        result = {"betas": beta_collection}
        for stat, data_list in stat_data.items():
            stat_collection = BrainCollection(
                data_list, mask=bc._mask, metadata=result_metadata, lazy=False
            )
            stat_collection._design_columns = design_columns
            result[stat] = stat_collection
        return result

    return beta_collection


def resolve_confounds(
    bc,
    confounds: str | list[pd.DataFrame | Path | str] | None,
) -> list[pd.DataFrame | Path | str] | None:
    """Resolve confounds argument to per-subject list.

    Args:
        bc: BrainCollection instance.
        confounds: Either:
            - str: Column name in metadata containing confound paths
            - list: Already per-subject list of DataFrames or paths
            - None: No confounds

    Returns:
        List of confounds (one per subject) or None
    """
    if confounds is None:
        return None

    if isinstance(confounds, str):
        # It's a metadata column name
        if confounds not in bc._metadata.columns:
            raise KeyError(
                f"Confounds column '{confounds}' not found in metadata. "
                f"Available: {list(bc._metadata.columns)}"
            )
        return list(bc._metadata[confounds])

    if isinstance(confounds, list):
        if len(confounds) != len(bc):
            raise ValueError(
                f"confounds list length ({len(confounds)}) must match "
                f"collection length ({len(bc)})"
            )
        return confounds

    raise TypeError(
        f"confounds must be str, list, or None, got {type(confounds).__name__}"
    )


def load_design_matrix(bc, path: str | Path) -> pd.DataFrame:
    """Load design matrix from a file path.

    Supports common formats: .csv, .tsv, .txt

    Args:
        bc: BrainCollection instance (unused, kept for API consistency).
        path: Path to design matrix file.

    Returns:
        DataFrame with design matrix contents.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Design matrix file not found: {path}")

    sep = "\t" if path.suffix in [".tsv", ".txt"] else ","
    return pd.read_csv(path, sep=sep)


def fit_ridge(
    bc,
    X: "np.ndarray | str | list",
    alpha: float | str = 1.0,
    cv: int | None = 5,
    scale: bool = True,
    scale_value: float = 100.0,
    output: str = "scores",
    save: dict[str, str] | None = None,
    show_progress: bool = True,
    **ridge_kwargs,
) -> "BrainCollection | dict[str, BrainCollection]":
    """
    Fit ridge regression to each subject in collection.

    Memory-efficient encoding model fitting that processes subjects one at a
    time. Default behavior returns cross-validated R-squared scores per voxel,
    suitable for group-level inference on encoding model performance.

    Args:
        bc: BrainCollection instance.
        X: Feature matrix. Can be:
            - np.ndarray: Shared features (n_samples, n_features) used for all subjects
            - str: Column name in metadata pointing to feature file paths
            - list: List of arrays/DataFrames, one per subject
        alpha: Ridge regularization parameter. Can be:
            - float: Fixed regularization strength
            - 'auto': Use cross-validation to select optimal alpha
        cv: Cross-validation folds for computing scores. Default is 5.
            Required when output='scores' or 'both'. Set to None only when
            output='weights'.
        scale: If True, apply percent signal change scaling before fitting.
        scale_value: Scaling value (default 100.0 for percent signal change).
        output: What to return. Options:
            - 'scores': CV R-squared scores per voxel (default, for encoding workflow)
            - 'weights': Model weights (n_features, n_voxels)
            - 'both': Dict with both 'scores' and 'weights'
        save: Dict mapping output type to path template, e.g.
            ``{'weights': 'output/{subject}_weights.nii.gz',
            'scores': 'output/{subject}_scores.nii.gz'}``.
            Supports {subject}, {session}, {idx}, and other metadata columns.
        show_progress: Show progress bar during fitting.
        **ridge_kwargs: Additional arguments passed to Ridge model
            (e.g., backend='torch', random_state=42).

    Returns:
        BrainCollection of scores or weights, or dict with both if output='both'.
        Each BrainData will have ``cv_results_`` attribute when cv is used.

    Examples:
        >>> # Encoding model workflow: get CV scores for group analysis
        >>> scores = bc.fit_ridge(X=features, alpha=1.0)
        >>> group_ttest = scores.ttest()  # Test encoding accuracy vs chance

        >>> # Get both scores and weights
        >>> results = bc.fit_ridge(X=features, alpha=1.0, output='both')
        >>> scores = results['scores']
        >>> weights = results['weights']

        >>> # Auto-select alpha with CV
        >>> scores = bc.fit_ridge(X=features, alpha='auto', cv=5)

        >>> # Get weights only (no CV needed)
        >>> weights = bc.fit_ridge(X=features, alpha=1.0, output='weights', cv=None)
    """
    from nltools.data.collection import BrainCollection
    from nltools.utils import attempt_to_import

    tqdm = attempt_to_import("tqdm", "tqdm")

    # Validate output parameter
    valid_outputs = {"scores", "weights", "both"}
    if output not in valid_outputs:
        raise ValueError(f"Invalid output: '{output}'. Valid options: {valid_outputs}")

    # CV is required for scores
    if output in ("scores", "both") and cv is None:
        raise ValueError(
            f"cv must be specified when output='{output}'. "
            "Set cv=5 (or another int) to compute cross-validated scores, "
            "or use output='weights' if you only need model weights."
        )

    # Resolve X to per-subject list (returns None if shared array)
    X_list = resolve_X(bc, X)

    # Progress bar setup
    iterator = range(len(bc))
    if show_progress and tqdm is not None:
        iterator = tqdm.tqdm(iterator, desc="Fitting Ridge", unit="subject")

    # Storage for results based on output type
    need_weights = output in ("weights", "both")
    need_scores = output in ("scores", "both")

    weight_data_list = [] if need_weights else None
    score_data_list = [] if need_scores else None
    result_metadata = []
    cv_results_list = []
    feature_names = None

    for i in iterator:
        # Load subject data
        bd = bc._load_item(i)
        metadata_row = bc._metadata.iloc[i]

        # Get subject features
        X_subj = X_list[i] if X_list else X

        # Load from file if needed
        if isinstance(X_subj, (str, Path)):
            X_subj = load_features(bc, X_subj)

        # Extract feature names if available
        if feature_names is None and hasattr(X_subj, "columns"):
            feature_names = list(X_subj.columns)

        # Apply scaling if requested
        if scale:
            bd = bd.scale(scale_value)

        # Fit ridge
        bd.fit(model="ridge", X=X_subj, alpha=alpha, cv=cv, **ridge_kwargs)

        result_metadata.append(metadata_row.to_dict())

        # Store CV results if available
        cv_result = None
        if hasattr(bd, "cv_results_") and bd.cv_results_ is not None:
            cv_result = bd.cv_results_
            cv_results_list.append(cv_result)

        # Extract weights if needed
        if need_weights:
            weights_data = bd.ridge_weights.data
            weights = bd[0].copy()
            weights.data = weights_data
            if feature_names:
                weights._feature_names = feature_names
            if cv_result:
                weights.cv_results_ = cv_result

            if save and "weights" in save:
                from nltools.data.collection import _resolve_save_path

                save_path = _resolve_save_path(save["weights"], metadata_row, i)
                weights.write(str(save_path))

            weight_data_list.append(weights)

        # Extract scores if needed
        if need_scores:
            scores_data = bd.ridge_scores.data
            scores = bd[0].copy()
            scores.data = scores_data
            if cv_result:
                scores.cv_results_ = cv_result

            if save and "scores" in save:
                from nltools.data.collection import _resolve_save_path

                save_path = _resolve_save_path(save["scores"], metadata_row, i)
                scores.write(str(save_path))

            score_data_list.append(scores)

        # Unload to free memory (only works for path-based collections)
        bc.unload([i])

    # Build result collection(s)
    result_meta_df = pd.DataFrame(result_metadata)

    if output == "weights":
        weight_collection = BrainCollection(
            weight_data_list,
            mask=bc.mask,
            metadata=result_meta_df,
        )
        if feature_names:
            weight_collection._feature_names = feature_names
        if cv_results_list:
            weight_collection.cv_results_ = cv_results_list
        return weight_collection

    elif output == "scores":
        score_collection = BrainCollection(
            score_data_list,
            mask=bc.mask,
            metadata=result_meta_df,
        )
        if cv_results_list:
            score_collection.cv_results_ = cv_results_list
        return score_collection

    else:  # output == "both"
        weight_collection = BrainCollection(
            weight_data_list,
            mask=bc.mask,
            metadata=result_meta_df,
        )
        if feature_names:
            weight_collection._feature_names = feature_names
        if cv_results_list:
            weight_collection.cv_results_ = cv_results_list

        score_collection = BrainCollection(
            score_data_list,
            mask=bc.mask,
            metadata=result_meta_df,
        )
        if cv_results_list:
            score_collection.cv_results_ = cv_results_list

        return {"weights": weight_collection, "scores": score_collection}


def resolve_X(
    bc,
    X: "np.ndarray | pd.DataFrame | str | list | None",
) -> list | None:
    """Resolve design/feature matrix X to per-subject list.

    Unified helper for resolving X parameter across fit methods. Supports
    three input patterns:
    1. Shared matrix (array/DataFrame/DesignMatrix): Same X for all subjects
    2. Per-subject list: List of matrices, one per subject
    3. Metadata column: String column name pointing to file paths

    Args:
        bc: BrainCollection instance.
        X: Design/feature matrix. Can be:
            - np.ndarray: Shared array (used for all subjects)
            - pd.DataFrame: Shared DataFrame/DesignMatrix (used for all subjects)
            - str: Column name in metadata containing file paths
            - list: Per-subject list of arrays/DataFrames/paths
            - None: Error

    Returns:
        list | None: Per-subject list if X varies by subject, None if shared.
            Caller should use: `X_subj = X_list[i] if X_list else X`
    """
    if X is None:
        raise ValueError("X must be provided")

    # Shared array - return None to signal no per-subject list
    if isinstance(X, np.ndarray):
        return None

    # Shared DataFrame - return None to signal shared
    if isinstance(X, pd.DataFrame):
        return None

    # Shared DesignMatrix (Polars-based, doesn't inherit from pd.DataFrame)
    from nltools.data.designmatrix import DesignMatrix

    if isinstance(X, DesignMatrix):
        return None

    # Metadata column name - return list of file paths
    if isinstance(X, str):
        if X not in bc._metadata.columns:
            raise KeyError(
                f"Column '{X}' not found in metadata. "
                f"Available: {list(bc._metadata.columns)}"
            )
        return list(bc._metadata[X])

    # Per-subject list - validate length
    if isinstance(X, list):
        if len(X) != len(bc):
            raise ValueError(
                f"X list length ({len(X)}) must match collection length ({len(bc)})"
            )
        return X

    raise TypeError(
        f"X must be np.ndarray, DataFrame, DesignMatrix, str, or list, "
        f"got {type(X).__name__}"
    )


def load_features(bc, path: str | Path) -> np.ndarray:
    """Load features from a file path.

    Supports common formats: .npy, .csv, .tsv, .txt

    Args:
        bc: BrainCollection instance (unused, kept for API consistency).
        path: Path to feature file.

    Returns:
        NumPy array of feature values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    elif suffix in [".csv", ".tsv", ".txt"]:
        sep = "\t" if suffix in [".tsv", ".txt"] else ","
        return pd.read_csv(path, sep=sep).values
    else:
        raise ValueError(f"Unsupported feature file format: {suffix}")
