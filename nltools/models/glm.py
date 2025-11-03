"""
GLM model for neuroimaging data.

Wraps nilearn.glm.first_level.FirstLevelModel with sklearn-compatible API.
"""

import numpy as np
import nibabel as nib
import warnings
from .base import BaseModel
from nilearn.glm.first_level import FirstLevelModel
from nltools.prefs import MNI_Template


class Glm(BaseModel):
    """
    General Linear Model for fMRI data analysis with sklearn-compatible API.

    Wraps nilearn.glm.first_level.FirstLevelModel using composition pattern,
    similar to how BrainData holds masker objects. Provides sklearn-style
    interface (fit/predict/score) while exposing full nilearn GLM functionality.

    Args:
        t_r (float, optional): Repetition time (TR) in seconds. If None, will be
            inferred from data.
        noise_model (str, default='ols'): Noise model for temporal autocorrelation
            ('ols' or 'ar1').

            - 'ols': Ordinary Least Squares (assumes independent errors)
            - 'ar1': Autoregressive AR(1) model (accounts for temporal correlation)

        smoothing_fwhm (float, optional): Full-Width at Half Maximum (FWHM) in mm
            for spatial smoothing. If None, no smoothing is applied.
        mask (Nifti1Image, optional): Mask image defining voxels to include in
            analysis. If None, uses MNI template mask (default, like BrainData).
        **kwargs: Additional arguments passed to nilearn FirstLevelModel.

    Attributes:
        is_fitted_ (bool): Whether the model has been fitted

    Note:
        Access fitted results via properties: ``glm_``, ``residuals``, ``design_matrices_``

    Examples:
        >>> from nltools.models import GLMModel
        >>> from nilearn.glm.first_level import make_first_level_design_matrix
        >>> import pandas as pd
        >>> import numpy as np
        >>> from nibabel import Nifti1Image
        >>>
        >>> # Create synthetic fMRI data
        >>> n_scans = 100
        >>> fmri_data = np.random.randn(n_scans, 20, 20, 20)
        >>> img = Nifti1Image(fmri_data.T, np.eye(4))
        >>>
        >>> # Create design matrix
        >>> frame_times = np.arange(n_scans) * 2.0
        >>> events = pd.DataFrame({
        ...     'onset': [10, 30, 50, 70],
        ...     'duration': [1, 1, 1, 1],
        ...     'trial_type': ['task', 'task', 'task', 'task']
        ... })
        >>> design_matrix = make_first_level_design_matrix(frame_times, events)
        >>>
        >>> # Fit GLM
        >>> model = GLMModel(t_r=2.0, noise_model='ar1')
        >>> model.fit(img, design_matrices=design_matrix)
        >>>
        >>> # Compute contrast
        >>> task_effect = model.compute_contrast('task', output_type='stat')
        >>>
        >>> # Get fitted values
        >>> fitted_values = model.predict()
        >>>
        >>> # Access residuals
        >>> residuals = model.residuals

    Notes:
        Unlike Ridge which works with 2D arrays (samples × features), GLMModel
        works with 4D neuroimaging data (x × y × z × time) and design matrices.
        Therefore, it does not use BaseModel's input validation methods.

        The predict() method follows sklearn's LinearRegression semantics:
        - predict() returns fitted values (predictions on training data)
        - predict(X) would generate predictions with new design matrix (future feature)

        For advanced use cases, access the internal FirstLevelModel via the
        ``glm_`` property to use any nilearn-specific functionality.
    """

    def __init__(
        self, t_r=None, noise_model="ols", smoothing_fwhm=None, mask=None, **kwargs
    ):
        # Initialize BaseModel
        super().__init__()

        # Store parameters
        self.t_r = t_r
        self.noise_model = noise_model
        self.smoothing_fwhm = smoothing_fwhm

        # Initialize mask (use MNI template if not provided, like BrainData)
        if mask is None:
            self.mask = nib.load(MNI_Template.mask)
        else:
            self.mask = mask

        # Compose FirstLevelModel (composition not inheritance)
        # Extract drift_model from kwargs if present, otherwise use None
        drift_model = kwargs.pop("drift_model", None)
        self._glm = FirstLevelModel(
            t_r=t_r,
            noise_model=noise_model,
            smoothing_fwhm=smoothing_fwhm,
            mask_img=self.mask,
            minimize_memory=False,  # Need this to access predictions
            standardize=False,  # User should standardize beforehand if needed
            drift_model=drift_model,  # Allow user to set, but warning will be suppressed when design matrices provided
            **kwargs,
        )

    def fit(self, X, y=None, design_matrices=None, events=None, **kwargs):
        """
        Fit GLM to fMRI data.

        Args:
            X (Nifti1Image or list of Nifti1Image): 4D fMRI image(s) to fit.
                Can be single run or list of runs.
            y (None): Not used, present for sklearn API compatibility.
            design_matrices (DataFrame, DesignMatrix, or list of DataFrame/DesignMatrix):
                Design matrix or list of design matrices (one per run). Each should
                have shape (n_scans, n_regressors). Accepts both pandas DataFrames
                and nltools DesignMatrix objects.
            events (DataFrame or list of DataFrame, optional): Event specifications
                for automatic design matrix creation. Alternative to providing
                design_matrices directly.
            **kwargs: Additional arguments passed to FirstLevelModel.fit()

        Returns:
            GLMModel: Fitted model instance (for method chaining)

        Notes:
            Unlike BaseModel's fit(), this method does not validate X as a 2D array
            because GLM works with 4D neuroimaging data. Input validation is
            delegated to nilearn's FirstLevelModel.

            DesignMatrix objects are automatically converted to pandas DataFrames
            for nilearn compatibility. The conversion is done at this boundary to
            keep DesignMatrix Polars-native while maintaining nilearn integration.
        """
        # Convert DesignMatrix to pandas for nilearn compatibility
        if design_matrices is not None:
            design_matrices_pd = self._convert_design_matrices(design_matrices)
        else:
            design_matrices_pd = None

        # Delegate to composed FirstLevelModel
        # Suppress warning about drift_model being ignored when design matrices are supplied
        # This is expected behavior since drift should be included in the design matrix
        with warnings.catch_warnings():
            # Filter the specific warning about drift_model being ignored
            # The warning message format: "If design matrices are supplied, [drift_model] will be ignored."
            # Use a permissive pattern to catch variations
            warnings.filterwarnings(
                "ignore",
                message=".*drift.*ignored.*|.*design matrices.*drift.*",
                category=UserWarning,
            )
            self._glm.fit(X, design_matrices=design_matrices_pd, events=events, **kwargs)

        # Set BaseModel fitted state
        self.is_fitted_ = True

        return self

    def _convert_design_matrices(self, design_matrices):
        """
        Convert DesignMatrix objects to pandas DataFrames for nilearn.

        Args:
            design_matrices (DesignMatrix, DataFrame, or list of either): Design
                matrix/matrices to convert

        Returns:
            DataFrame or list of DataFrame: Pandas DataFrames for nilearn consumption
        """
        # Import here to avoid circular dependency
        from nltools.data import DesignMatrix

        # Handle single design matrix
        if not isinstance(design_matrices, list):
            if isinstance(design_matrices, DesignMatrix):
                return design_matrices._to_pandas()
            else:
                return design_matrices

        # Handle list of design matrices
        converted = []
        for dm in design_matrices:
            if isinstance(dm, DesignMatrix):
                converted.append(dm._to_pandas())
            else:
                converted.append(dm)

        return converted

    def predict(self, X=None):
        """
        Generate predictions from fitted GLM.

        Args:
            X (DataFrame or None, default=None): Design matrix for generating
                predictions.

                - If None: returns fitted values (predictions on training data)
                - If DataFrame: generates predictions using new design matrix
                  (not yet implemented)

        Returns:
            list of Nifti1Image: Predicted brain activity for each run

        Notes:
            Follows sklearn's LinearRegression semantics where predict() without
            arguments returns fitted values (like calling predict(X_train)).

            Future enhancement will support predict(X=new_design_matrix) to
            generate predictions with different experimental designs.
        """
        self._check_is_fitted()

        if X is None:
            # Return fitted values (predictions on training data)
            return self._glm.predicted
        else:
            # Future: Generate predictions with new design matrix
            # Would compute: predicted_data = betas @ X.T at each voxel
            raise NotImplementedError(
                "Prediction with new design matrix not yet implemented. "
                "Use predict() without arguments to get fitted values."
            )

    def score(self, X=None, y=None):
        """
        Return mean R² across voxels and runs.

        Computes average coefficient of determination (R²) from the fitted GLM.
        Higher values indicate better model fit.

        Args:
            X (None): Not used, present for sklearn API compatibility.
            y (None): Not used, present for sklearn API compatibility.

        Returns:
            float: Mean R² across all voxels and runs. Range: [0, 1], higher is better.

        Notes:
            Extracts R² values from nilearn's FirstLevelModel.r_square attribute,
            which returns a list of Nifti1Image objects (one per run).
            Computes the mean across all non-NaN voxels and all runs.

            For voxel-wise R² maps, access `glm_.r_square` directly.

        Examples:
            >>> brain.fit(model='glm', X=design_matrix)
            >>> r2 = brain.model_.score()
            >>> print(f"Mean R²: {r2:.3f}")
        """
        self._check_is_fitted()

        # Get R² maps from nilearn (list of Nifti1Image objects, one per run)
        r_square_maps = self._glm.r_square

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

    def compute_contrast(self, contrast_def, output_type="stat"):
        """
        Compute contrast using nilearn for accurate statistical inference.

        This is the primary method for extracting results from a fitted GLM.
        Delegates to nilearn's FirstLevelModel.compute_contrast() for proper
        statistical inference with correct degrees of freedom, etc.

        Args:
            contrast_def (str, array-like, or dict): Contrast specification:
                - str: Regressor name (e.g., 'task')
                - array-like: Contrast vector (e.g., [1, -1, 0, 0])
                - dict: Multiple contrasts with names as keys
            output_type (str, default='stat'): Type of output to return:
                - 'stat': T-statistic map (default)
                - 'z_score': Z-score map
                - 'p_value': P-value map
                - 'effect_size': Effect size (beta) map
                - 'effect_variance': Variance of effect size
                - 'all': Dictionary with all output types

        Returns:
            Nifti1Image or dict: Contrast map(s). If output_type='all', returns dict
                with all maps.

        Examples:
            >>> # After fitting model
            >>> model.fit(img, design_matrices=design_matrix)
            >>>
            >>> # Simple contrast by name
            >>> t_map = model.compute_contrast('task')
            >>>
            >>> # Contrast vector
            >>> contrast_map = model.compute_contrast([1, -1, 0])
            >>>
            >>> # Get all outputs
            >>> results = model.compute_contrast('task', output_type='all')
            >>> t_map = results['stat']
            >>> p_map = results['p_value']
        """
        self._check_is_fitted()
        return self._glm.compute_contrast(contrast_def, output_type=output_type)

    # Properties for accessing FirstLevelModel attributes (advanced use)

    @property
    def residuals(self):
        """
        Residuals from fitted GLM.

        Returns:
            list of Nifti1Image: Residual images for each run (observed - predicted)

        Raises:
            ValueError: If model has not been fitted yet
        """
        self._check_is_fitted()
        return self._glm.residuals

    @property
    def design_matrices_(self):
        """
        Design matrices used in fitting.

        Returns:
            list of DataFrame: Design matrices for each run

        Raises:
            ValueError: If model has not been fitted yet
        """
        self._check_is_fitted()
        return self._glm.design_matrices_

    @property
    def glm_(self):
        """
        Access internal FirstLevelModel for advanced use.

        Provides direct access to the wrapped nilearn FirstLevelModel
        instance for advanced users who need functionality not exposed
        by the sklearn-compatible interface.

        Returns:
            FirstLevelModel: Internal nilearn FirstLevelModel instance

        Examples:
            >>> # Access nilearn-specific attributes
            >>> model.glm_.labels_
            >>> model.glm_.results_
            >>>
            >>> # Use nilearn-specific methods
            >>> model.glm_.generate_report()
        """
        return self._glm
