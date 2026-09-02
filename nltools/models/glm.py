"""GLM model for neuroimaging data.

Wraps nilearn.glm.first_level.FirstLevelModel with sklearn-compatible API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import nibabel as nib
import warnings
from .base import BaseModel
from nilearn.glm.first_level import FirstLevelModel
from nltools.templates import get_brainspace

if TYPE_CHECKING:
    import pandas as pd

    from nltools.data import DesignMatrix


class Glm(BaseModel):
    """General Linear Model for fMRI data analysis with sklearn-compatible API.

    Wraps `nilearn.glm.first_level.FirstLevelModel` by composition, similar to
    how `BrainData` holds masker objects. Provides the sklearn-style
    fit/predict/score interface while exposing full nilearn GLM functionality
    through the `glm_` property.

    Unlike `Ridge`, which works with 2-D arrays (samples × features), `Glm`
    works with 4-D neuroimaging data (x × y × z × time) and design matrices, so
    it does not use `BaseModel`'s input validation. `predict()` follows sklearn's
    `LinearRegression` semantics: with no argument it returns the fitted values
    on the training data; with a new design matrix it returns `X @ coef_`
    (single-run fits only).

    Args:
        t_r (float, optional): Repetition time (TR) in seconds. If None, inferred
            from the data.
        noise_model (str): Noise model for temporal autocorrelation: `'ols'`
            (ordinary least squares, independent errors) or `'ar1'` (autoregressive
            AR(1), accounts for temporal correlation). Default `'ols'`.
        smoothing_fwhm (float, optional): Full width at half maximum in mm for
            spatial smoothing. If None, no smoothing is applied.
        mask (nibabel.Nifti1Image, optional): Mask defining the voxels to analyze.
            If None, uses the package brain-space mask (like `BrainData`).
        progress_bar (bool): If True, enable nilearn's per-run progress output.
            Default False.
        **kwargs (dict): Forwarded to `nilearn.glm.first_level.FirstLevelModel`
            (e.g. `drift_model`, `hrf_model`, `memory`).

    Attributes:
        is_fitted_ (bool): Whether the model has been fitted.
        coef_ (np.ndarray | list[np.ndarray]): Beta matrix `(n_regressors, n_voxels)`
            after fitting a single run, or one per run for multi-run fits.
        mask (nibabel.Nifti1Image): Mask image used for analysis.
        glm_ (FirstLevelModel): The wrapped nilearn model, for advanced use.
        residuals (list[nibabel.Nifti1Image]): Residual images, one per run.
        design_matrices_ (list[pd.DataFrame]): Design matrices used in fitting, one
            per run.

    Examples:
        ```python
        import numpy as np
        import pandas as pd
        from nibabel import Nifti1Image
        from nilearn.glm.first_level import make_first_level_design_matrix
        from nltools.models import Glm

        # Synthetic fMRI data and a matching design matrix
        n_scans = 100
        img = Nifti1Image(np.random.randn(20, 20, 20, n_scans), np.eye(4))
        frame_times = np.arange(n_scans) * 2.0
        events = pd.DataFrame(
            {"onset": [10, 30, 50, 70], "duration": [1, 1, 1, 1], "trial_type": ["task"] * 4}
        )
        design_matrix = make_first_level_design_matrix(frame_times, events)

        model = Glm(t_r=2.0, noise_model="ar1")
        model.fit(img, design_matrices=design_matrix)

        task_effect = model.compute_contrast("task", output_type="stat")
        fitted_values = model.predict()
        residuals = model.residuals
        ```
    """

    def __init__(
        self,
        *,
        t_r: float | None = None,
        noise_model: str = "ols",
        smoothing_fwhm: float | None = None,
        mask: nib.Nifti1Image | None = None,
        progress_bar: bool = False,
        **kwargs,
    ) -> None:
        # Initialize BaseModel
        super().__init__()

        # Store parameters
        self.t_r = t_r
        self.noise_model = noise_model
        self.smoothing_fwhm = smoothing_fwhm
        self.progress_bar = progress_bar

        # Initialize mask (use MNI template if not provided, like BrainData)
        if mask is None:
            self.mask = nib.load(get_brainspace().mask)
        else:
            self.mask = mask

        # Compose FirstLevelModel (composition not inheritance)
        # Extract drift_model from kwargs if present, otherwise use None
        drift_model = kwargs.pop("drift_model", None)
        # Route progress_bar to nilearn's real per-run verbose output
        kwargs.setdefault("verbose", 1 if progress_bar else 0)
        self._glm = FirstLevelModel(
            t_r=t_r,
            noise_model=noise_model,
            smoothing_fwhm=smoothing_fwhm,
            mask_img=self._build_masker(smoothing_fwhm, kwargs),
            minimize_memory=False,  # Need this to access predictions
            standardize=False,  # User should standardize beforehand if needed
            signal_scaling=False,  # Scaling is owned by BrainData.fit's explicit
            # scale/standardize pipeline, not inherited from nilearn's default.
            drift_model=drift_model,  # Allow user to set, but warning will be suppressed when design matrices provided
            **kwargs,
        )

    def _build_masker(self, smoothing_fwhm, glm_kwargs):
        """Pre-fit a ``NiftiMasker`` on ``self.mask`` for ``FirstLevelModel``.

        Handing nilearn a bare ``Nifti1Image`` as ``mask_img`` makes it build a
        ``MultiNiftiMasker`` and fit it on the run images, which emits a
        ``RuntimeWarning`` ("Generation of a mask has been requested ... while
        a mask was given") on every fit even though the given mask is what gets
        used. A fitted masker is used directly (``FirstLevelModel._prepare_mask``
        sets ``masker_ = mask_img``) with identical betas. nilearn does not copy
        its own masker-relevant parameters onto a user-supplied masker, so every
        one ``FirstLevelModel`` would otherwise forward is set here from Glm's
        values (``standardize`` is overridden by nilearn after the fact).
        """
        from nilearn.maskers import NiftiMasker

        masker = NiftiMasker(
            mask_img=self.mask,
            smoothing_fwhm=smoothing_fwhm,
            t_r=self.t_r,
            target_affine=glm_kwargs.get("target_affine"),
            target_shape=glm_kwargs.get("target_shape"),
            memory=glm_kwargs.get("memory"),
            # nilearn's own embedded masker runs one level quieter than the GLM.
            memory_level=max(0, glm_kwargs.get("memory_level", 1) - 1),
            verbose=max(0, glm_kwargs.get("verbose", 0) - 1),
        )
        return masker.fit()

    def fit(  # nosemgrep: kwargs-internal-forwarding  # forwards to nilearn FirstLevelModel.fit
        self,
        X: nib.Nifti1Image | list[nib.Nifti1Image],
        y: None = None,
        *,
        design_matrices: pd.DataFrame
        | DesignMatrix
        | list[pd.DataFrame | DesignMatrix]
        | None = None,
        events: pd.DataFrame | list[pd.DataFrame] | None = None,
        **kwargs,
    ) -> Glm:
        """Fit GLM to fMRI data.

        Args:
            X (nibabel.Nifti1Image | list[nibabel.Nifti1Image]): 4-D fMRI image(s) to
                fit, a single run or a list of runs.
            y (None): Not used; present for sklearn API compatibility.
            design_matrices (pd.DataFrame | DesignMatrix | list, optional): Design
                matrix or one per run, each of shape `(n_scans, n_regressors)`.
                `DesignMatrix` objects are converted to pandas at this boundary.
            events (pd.DataFrame | list[pd.DataFrame], optional): Event
                specifications for automatic design-matrix creation; an alternative
                to `design_matrices`.
            **kwargs (dict): Forwarded to `FirstLevelModel.fit`.

        Returns:
            Glm: The fitted model (for method chaining).

        Note:
            Unlike `BaseModel.fit`, this method does not validate `X` as a 2-D array
            because the GLM works with 4-D neuroimaging data; validation is
            delegated to nilearn's `FirstLevelModel`.
        """
        # Convert DesignMatrix to pandas for nilearn compatibility
        if design_matrices is not None:
            design_matrices_pd = self._convert_design_matrices(design_matrices)
        else:
            design_matrices_pd = None

        # Delegate to composed FirstLevelModel. Progress output is driven by
        # nilearn's verbose parameter (set in __init__ from progress_bar), which
        # reports real per-run progress instead of a synthetic 3-step bar.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*drift.*ignored.*|.*design matrices.*drift.*",
                category=UserWarning,
            )
            self._glm.fit(
                X, design_matrices=design_matrices_pd, events=events, **kwargs
            )

        # Set BaseModel fitted state
        self.is_fitted_ = True

        # Cache the beta matrix so predict(X) mirrors Ridge (X @ coef_).
        self._extract_coef()

        return self

    def _extract_coef(self) -> None:
        """Cache ``coef_`` (betas) from the fitted run_glm results.

        nilearn stores the GLS parameter estimates as ``theta`` on each
        per-AR-bin ``RegressionResults`` (keyed by ``labels_``); assembling
        them across bins yields the full ``(n_regressors, n_voxels)`` beta
        matrix directly from arrays — no unmasking to a Nifti. Sets ``coef_``
        to that 2-D array for a single-run fit, or a list of them per run.
        """
        coefs = []
        for run in range(len(self._glm.labels_)):
            labels = self._glm.labels_[run]
            results = self._glm.results_[run]
            n_reg = self._glm.design_matrices_[run].shape[1]
            beta = np.zeros((n_reg, labels.shape[0]))
            for lab in results:
                beta[:, labels == lab] = results[lab].theta
            coefs.append(beta)
        self.coef_ = coefs[0] if len(coefs) == 1 else coefs

    def _convert_design_matrices(self, design_matrices):
        """Convert DesignMatrix objects to pandas DataFrames for nilearn.

        Args:
            design_matrices (DesignMatrix, DataFrame, or list of either): Design
                matrix/matrices to convert

        Returns:
            A pandas DataFrame (or list of them) for nilearn consumption.
        """
        # Import here to avoid circular dependency
        from nltools.data import DesignMatrix

        # Handle single design matrix
        if not isinstance(design_matrices, list):
            if isinstance(design_matrices, DesignMatrix):
                return design_matrices.to_pandas()
            return design_matrices

        # Handle list of design matrices
        converted = []
        for dm in design_matrices:
            if isinstance(dm, DesignMatrix):
                converted.append(dm.to_pandas())
            else:
                converted.append(dm)

        return converted

    def predict(
        self, X: np.ndarray | pd.DataFrame | None = None
    ) -> list[nib.Nifti1Image] | np.ndarray:
        """Predict from the fitted GLM.

        With `X=None`, returns the fitted values on the training data (one
        `Nifti1Image` per run), matching sklearn's `LinearRegression` semantics.
        With a new design matrix, returns `X @ coef_` as a 2-D array, mirroring
        `Ridge.predict`; this requires a single-run fit.

        Args:
            X (np.ndarray | pd.DataFrame, optional): New design matrix of shape
                `(n_samples, n_regressors)`. Default None.

        Returns:
            list[nibabel.Nifti1Image] | np.ndarray: Fitted images per run when `X`
                is None; otherwise predictions of shape `(n_samples, n_voxels)`.

        Raises:
            NotImplementedError: If X is given for a multi-run fit (a single new
                design is ambiguous across runs — fit per run instead).
            ValueError: If X's column count does not match the fitted design.
        """
        self._check_is_fitted()

        if X is None:
            return self._glm.predicted_

        if isinstance(self.coef_, list):
            raise NotImplementedError(
                "predict(X) with a new design is only supported for single-run "
                "GLM fits; this model was fit on multiple runs. Fit each run "
                "separately to predict from a new design."
            )

        # Accept pandas/polars DataFrames (both expose to_numpy) or raw arrays;
        # getattr keeps the callable check off the typed ndarray branch.
        to_numpy = getattr(X, "to_numpy", None)
        X = to_numpy() if to_numpy is not None else np.asarray(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} columns but the model was fit with "
                f"{self.coef_.shape[0]} regressors."
            )
        return X @ self.coef_

    def report(  # nosemgrep: kwargs-internal-forwarding  # forwards to nilearn generate_report
        self, contrasts=None, **kwargs
    ):
        """Generate a nilearn HTML report for the fitted GLM.

        Delegates to the underlying `FirstLevelModel.generate_report`, which
        renders the design matrix, requested contrast maps, and model
        parameters as a self-contained HTML report.

        Args:
            contrasts (str | list | dict, optional): Contrast(s) to render, in the
                same forms as `compute_contrast`.
            **kwargs (dict): Forwarded to nilearn's `generate_report` (e.g. `title`,
                `threshold`, `alpha`).

        Returns:
            HTMLReport: nilearn report object; call `.save_as_html(path)` or
                display it in a notebook.
        """
        self._check_is_fitted()
        return self._glm.generate_report(contrasts=contrasts, **kwargs)

    def score(self, X: None = None, y: None = None) -> float:
        """Return mean R² across voxels and runs.

        Computes average coefficient of determination (R²) from the fitted GLM.
        Higher values indicate better model fit.

        Args:
            X (None): Not used; present for sklearn API compatibility.
            y (None): Not used; present for sklearn API compatibility.

        Returns:
            float: Mean R² across all non-NaN voxels and all runs, in `[0, 1]`.

        Note:
            Averages nilearn's per-run `r_square_` maps. For voxel-wise R² maps,
            access `glm_.r_square_` directly.

        Examples:
            ```python
            brain.fit(model="glm", X=design_matrix)
            r2 = brain.model_.score()
            ```
        """
        self._check_is_fitted()

        # Get R² maps from nilearn (list of Nifti1Image objects, one per run)
        r_square_maps = self._glm.r_square_

        if r_square_maps is None or len(r_square_maps) == 0:
            raise ValueError(
                "R² maps not available. Ensure the model has been fitted successfully."
            )

        # Extract data arrays and compute mean across voxels and runs
        r_square_values = []
        for r2_img in r_square_maps:
            r2_data = r2_img.get_fdata()
            # Only include non-NaN voxels (voxels outside mask will be NaN)
            valid_voxels = r2_data[~np.isnan(r2_data)]
            if len(valid_voxels) > 0:
                r_square_values.append(valid_voxels)

        if len(r_square_values) == 0:
            raise ValueError(
                "No valid R² values found. All voxels are NaN. "
                "Check that the GLM fit completed successfully."
            )

        # Concatenate all runs and compute overall mean
        all_r_square = np.concatenate(r_square_values)
        mean_r_square = np.mean(all_r_square)

        return float(mean_r_square)

    def compute_contrast(
        self,
        contrast_def: str | np.ndarray | list | dict,
        output_type: str = "stat",
    ) -> nib.Nifti1Image | dict:
        """Compute a contrast using nilearn's statistical inference.

        This is the primary method for extracting results from a fitted GLM.
        Delegates to `FirstLevelModel.compute_contrast` for inference with the
        correct degrees of freedom.

        Args:
            contrast_def (str | np.ndarray | list | dict): A regressor name (e.g.
                `'task'`), a contrast vector (e.g. `[1, -1, 0, 0]`), or a dict of
                named contrasts.
            output_type (str): `'stat'` (t-statistic map, default), `'z_score'`,
                `'p_value'` (one-sided, per the nilearn/SPM directional-contrast
                convention; flip the contrast for the other direction),
                `'effect_size'` (beta), `'effect_variance'`, or `'all'` (a dict of
                every map).

        Returns:
            nibabel.Nifti1Image | dict: The contrast map, or a dict of all maps keyed
                by output type when `output_type='all'`.

        Examples:
            ```python
            model.fit(img, design_matrices=design_matrix)

            t_map = model.compute_contrast("task")  # by regressor name
            contrast_map = model.compute_contrast([1, -1, 0])  # contrast vector

            results = model.compute_contrast("task", output_type="all")
            t_map, p_map = results["stat"], results["p_value"]
            ```
        """
        self._check_is_fitted()
        return self._glm.compute_contrast(contrast_def, output_type=output_type)

    # Properties for accessing FirstLevelModel attributes (advanced use)

    @property
    def residuals(self) -> list[nib.Nifti1Image]:
        """Residuals from the fitted GLM.

        Returns:
            list[nibabel.Nifti1Image]: Residual images (observed − predicted), one
                per run.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        self._check_is_fitted()
        return self._glm.residuals_

    @property
    def design_matrices_(self) -> list[pd.DataFrame]:
        """Design matrices used in fitting.

        Returns:
            list[pd.DataFrame]: Design matrices, one per run.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        self._check_is_fitted()
        return self._glm.design_matrices_

    @property
    def glm_(self) -> FirstLevelModel:  # ty: ignore[invalid-type-form]  # ty>=0.0.61 mis-infers nilearn FirstLevelModel (a real class) as callable
        """Access the wrapped nilearn `FirstLevelModel` for advanced use.

        Exposes functionality not covered by the sklearn-compatible interface.

        Returns:
            FirstLevelModel: The internal nilearn model instance.

        Examples:
            ```python
            model.glm_.labels_  # nilearn-specific attributes
            model.glm_.results_
            model.glm_.generate_report()  # nilearn-specific methods
            ```
        """
        return self._glm
