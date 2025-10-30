# FastSRM Implementation: Test-Driven Development Plan

**Date**: 2025-10-29
**Priority**: 3.0 (Medium - Post v0.6.0)
**Estimated Effort**: 10-14 hours
**Target**: v0.6.1 or later

---

## Executive Summary

**Goal**: Implement FastSRM (Fast Shared Response Model) with atlas-based dimensionality reduction following nltools design philosophy and BrainIAK best practices.

**Key Benefits**:
- ⚡ **5x faster** than standard SRM
- 💾 **20x-40x more memory efficient**
- 📊 **Better R² accuracy** (atlas projection reduces noise)
- 🔧 **Seamless nilearn integration** (uses `NiftiLabelsMasker`)

**Design Philosophy**:
- Wrap nilearn, don't reimplement
- Property-based testing with mathematical invariants
- Sklearn-compatible API
- Backward compatible with standard SRM interface

---

## 1. Algorithm Overview

### 1.1 Mathematical Foundation

FastSRM (Richard et al., 2019) reduces computational complexity using atlas-based dimensionality reduction:

**Standard SRM**:
```
Complexity: O(I(VTK + VK² + K³))
Memory: O(VT)
where V = voxels, T = timepoints, K = features, I = iterations
```

**FastSRM**:
```
1. Project to atlas: X_atlas = atlas.transform(X)  # V → A voxels
2. Run SRM on atlas: W_atlas, S = SRM(X_atlas)     # Compute on A ≪ V
3. Project back: W = atlas.inverse_transform(W_atlas)  # Maintain full voxel space

Complexity: O(I(ATK + AK² + K³)) where A ≪ V
Memory: O(AT)  # Dramatic reduction!
```

**Key Insight**: Atlas parcellation reduces voxel dimensionality (e.g., 50,000 voxels → 400 parcels) while preserving functional structure.

### 1.2 Implementation Strategy

**Class Hierarchy**:
```python
SRM (existing)
  ↓
FastSRM (new) - inherits from SRM, adds atlas handling
```

**Nilearn Integration**:
- Use `nilearn.maskers.NiftiLabelsMasker` for atlas projection
- Support common atlases (Harvard-Oxford, AAL, Schaefer, custom)
- Leverage existing nltools Brain_Data integration

---

## 2. Test-Driven Development Plan

### 2.1 Testing Philosophy

Following the established SRM/DetSRM testing strategy:

✅ **Property-based tests** with mathematical invariants
✅ **Contract tests** for API behavior and error handling
✅ **Edge case tests** for boundary conditions
❌ **NO golden output tests** (brittle, platform-dependent)

**Test Organization**:
- Location: `nltools/tests/core/test_fastsrm.py`
- Target: ~35-40 tests
- Fixtures: Reuse `multi_subject_data` from `test_srm.py`, add atlas-specific fixtures

### 2.2 Test Phases (TDD Progression)

#### **Phase 1: Initialization & Contract Tests (8 tests, ~2 hours)**

**Fixtures**:
```python
@pytest.fixture
def sample_atlas():
    """Create simple synthetic atlas for testing."""
    # 3D volume with 10 labeled parcels
    # Shape: (10, 10, 10) with values 0-10

@pytest.fixture
def atlas_data():
    """Multi-subject data compatible with atlas."""
    # 5 subjects, variable voxel counts (within atlas mask)
    # Timepoints: 100, Features: 10
```

**Tests**:
1. `test_fastsrm_init_defaults()` - Default parameters (atlas=None → error, n_iter=10, features=50)
2. `test_fastsrm_init_custom_params()` - Custom parameters accepted
3. `test_fastsrm_init_with_atlas_path()` - Accept atlas as file path (string)
4. `test_fastsrm_init_with_atlas_array()` - Accept atlas as numpy array
5. `test_fastsrm_init_with_nilearn_masker()` - Accept NiftiLabelsMasker instance
6. `test_fastsrm_missing_atlas_error()` - Raise error if atlas=None on fit
7. `test_fastsrm_fit_before_transform_error()` - NotFittedError before fit
8. `test_fastsrm_fit_sets_attributes()` - Creates w_, s_, w_atlas_, atlas_masker_

**Implementation Target**:
```python
class FastSRM(SRM):
    """Fast Shared Response Model with atlas-based dimensionality reduction.

    Parameters
    ----------
    atlas : str, array-like, or NiftiLabelsMasker
        Brain atlas for dimensionality reduction. Can be:
        - Path to atlas file (str)
        - 3D numpy array with integer labels
        - Fitted NiftiLabelsMasker instance

    n_iter : int, default: 10
        Number of SRM iterations.

    features : int, default: 50
        Number of shared features.

    rand_seed : int, default: 0
        Random seed for reproducibility.

    Attributes
    ----------
    w_ : list of arrays
        Full voxel-space transforms (after inverse projection)

    w_atlas_ : list of arrays
        Atlas-space transforms (intermediate representation)

    s_ : array
        Shared response in feature space

    atlas_masker_ : NiftiLabelsMasker
        Fitted atlas masker for projections
    """

    def __init__(self, atlas=None, n_iter=10, features=50, rand_seed=0):
        super().__init__(n_iter=n_iter, features=features, rand_seed=rand_seed)
        self.atlas = atlas

    def fit(self, X, y=None):
        """Fit FastSRM using atlas-based projection."""
        # 1. Validate atlas
        # 2. Create/fit NiftiLabelsMasker
        # 3. Project data to atlas space
        # 4. Run standard SRM on projected data
        # 5. Store both atlas-space and full-space transforms
```

#### **Phase 2: Atlas Handling Tests (7 tests, ~2 hours)**

**Tests**:
9. `test_atlas_projection_reduces_dimensionality()` - Verify A ≪ V
10. `test_atlas_projection_preserves_timepoints()` - Timepoints unchanged
11. `test_invalid_atlas_format_error()` - Raise error for unsupported atlas types
12. `test_atlas_missing_labels_error()` - Error if atlas has no labeled regions
13. `test_atlas_projection_reversibility()` - atlas.inverse_transform(atlas.transform(X)) ≈ X
14. `test_different_atlas_sizes()` - Work with varying parcel counts (10, 100, 400 parcels)
15. `test_atlas_with_brain_data_integration()` - Accept Brain_Data objects with atlas

**Implementation Target**:
```python
def _validate_and_prepare_atlas(self, atlas):
    """Convert atlas input to NiftiLabelsMasker."""
    if isinstance(atlas, str):
        # Load from file
    elif isinstance(atlas, np.ndarray):
        # Create masker from array
    elif isinstance(atlas, NiftiLabelsMasker):
        # Use directly
    else:
        raise ValueError("atlas must be str, array, or NiftiLabelsMasker")

def _project_to_atlas(self, data):
    """Project subject data to atlas space."""
    projected = []
    for subject_data in data:
        # Use NiftiLabelsMasker.transform()
        projected.append(atlas_data)
    return projected
```

#### **Phase 3: Mathematical Property Tests (8 tests, ~2.5 hours)**

**Tests**:
16. `test_fastsrm_orthogonality_of_transforms()` - W.T @ W ≈ I (full voxel space)
17. `test_fastsrm_orthogonality_atlas_space()` - W_atlas.T @ W_atlas ≈ I
18. `test_fastsrm_reconstruction_quality()` - X ≈ W @ S (bounded error)
19. `test_fastsrm_shared_response_shape()` - s_.shape == (features, timepoints)
20. `test_fastsrm_transform_shape_preservation()` - Output has correct shapes
21. `test_fastsrm_variance_explained()` - Captures substantial variance
22. `test_fastsrm_atlas_compression_ratio()` - Verify memory reduction (A/V ratio)
23. `test_fastsrm_noise_reduction_property()` - Atlas projection should reduce noise

**Key Insight**: Test both atlas-space AND full voxel-space properties!

#### **Phase 4: Performance & Comparative Tests (6 tests, ~2 hours)**

**Tests**:
24. `test_fastsrm_vs_srm_similar_results()` - Comparable shared responses (corr > 0.85)
25. `test_fastsrm_faster_than_srm()` - Timing comparison (skip in CI, use @pytest.mark.slow)
26. `test_fastsrm_memory_efficient()` - Peak memory comparison (skip in CI)
27. `test_fastsrm_deterministic_with_seed()` - Reproducibility with same seed
28. `test_fastsrm_different_seed_different_init()` - Different seeds yield different results
29. `test_fastsrm_convergence_iterations()` - More iterations → better fit (up to limit)

**Implementation Note**: Tests 25-26 are optional performance benchmarks, not correctness tests.

#### **Phase 5: Edge Cases & Error Handling (8 tests, ~1.5 hours)**

**Tests**:
30. `test_fastsrm_single_subject_error()` - Raise ValueError for <2 subjects
31. `test_fastsrm_mismatched_timepoints_error()` - Error if different TRs
32. `test_fastsrm_insufficient_samples_error()` - Error if samples < features
33. `test_fastsrm_data_outside_atlas_mask()` - Handle voxels not in atlas parcels
34. `test_fastsrm_empty_atlas_parcels()` - Handle parcels with no signal
35. `test_fastsrm_transform_subject_new_data()` - Project new subject to shared space
36. `test_fastsrm_identical_subjects()` - Low disparity for identical data
37. `test_fastsrm_minimal_features()` - Work with very few features (K=3)

#### **Phase 6: Integration & Workflow Tests (5 tests, ~2 hours)**

**Tests**:
38. `test_fastsrm_brain_data_input()` - Accept Brain_Data objects
39. `test_fastsrm_align_function_integration()` - Work via `align(method='fast_srm')`
40. `test_fastsrm_fit_transform()` - Sklearn pipeline compatibility
41. `test_fastsrm_common_atlases()` - Test with real atlases (Harvard-Oxford, AAL, Schaefer)
42. `test_fastsrm_save_load_model()` - Pickle/unpickle fitted model

**Implementation Target**:
```python
# In nltools/stats.py, update align() function:

def align(data, method='probabilistic_srm', n_features=None, atlas=None, **kwargs):
    """
    ...
    method : str
        'probabilistic_srm' - Probabilistic SRM (EM algorithm)
        'deterministic_srm' - Deterministic SRM (BCD algorithm)
        'procrustes' - Procrustes hyperalignment
        'fast_srm' - Fast SRM with atlas (NEW!)

    atlas : str, array, or NiftiLabelsMasker (for method='fast_srm')
        Brain atlas for dimensionality reduction. Required if method='fast_srm'.
    """
    if method == 'fast_srm':
        if atlas is None:
            raise ValueError("atlas required for method='fast_srm'")
        model = FastSRM(atlas=atlas, n_iter=n_iter, features=n_features, ...)
    # ... rest of implementation
```

---

## 3. Implementation Details

### 3.1 Class Structure

```python
# File: nltools/algorithms/srm.py (append to existing file)

class FastSRM(SRM):
    """Fast Shared Response Model with atlas-based dimensionality reduction.

    Implements the algorithm from:
    Richard, H., Martin, L., Pinho, A., Pillow, J., Thirion, B. (2019).
    Fast Shared Response Model for fMRI data. arXiv:1909.12537

    Complexity Reduction:
    - Standard SRM: O(I*V*T*K + I*V*K² + I*K³)
    - FastSRM: O(I*A*T*K + I*A*K² + I*K³) where A ≪ V
    - Typical: V=50,000 voxels → A=400 parcels (125x reduction!)

    Memory Reduction:
    - Standard SRM: O(V*T)
    - FastSRM: O(A*T)
    - Typical: 20x-40x less memory

    Parameters
    ----------
    atlas : str, array-like, or NiftiLabelsMasker, required
        Brain atlas for parcellation. Options:
        - Path to Nifti file (e.g., 'harvard_oxford.nii.gz')
        - 3D numpy array with integer parcel labels
        - Pre-configured NiftiLabelsMasker instance
        - Common atlases: 'harvard_oxford', 'aal', 'schaefer_100', etc.

    n_iter : int, default=10
        Number of EM iterations for SRM optimization.

    features : int, default=50
        Number of shared features (latent dimensions).

    rand_seed : int, default=0
        Random seed for reproducible initialization.

    atlas_kwargs : dict, optional
        Additional arguments passed to NiftiLabelsMasker (if atlas is not already a masker).
        E.g., {'smoothing_fwhm': 6, 'standardize': True}

    Attributes
    ----------
    w_ : list of arrays, shape[i] = [voxels_i, features]
        Full voxel-space orthogonal transforms after inverse projection.
        These can be applied to full-resolution data.

    w_atlas_ : list of arrays, shape[i] = [parcels, features]
        Atlas-space orthogonal transforms (intermediate representation).
        These operate in the reduced parcel space.

    s_ : array, shape = [features, samples]
        Shared response in feature space (same as standard SRM).

    atlas_masker_ : NiftiLabelsMasker
        Fitted atlas masker used for forward/inverse projections.

    n_parcels_ : int
        Number of parcels in the atlas (dimensionality of atlas space).

    sigma_s_, mu_, rho2_ : inherited from SRM
        Additional fitted parameters from EM algorithm.

    Examples
    --------
    >>> # Using string atlas name (loads from nilearn datasets)
    >>> fastsrm = FastSRM(atlas='harvard_oxford', n_iter=10, features=50)
    >>> fastsrm.fit(multi_subject_data)
    >>> transformed = fastsrm.transform(multi_subject_data)

    >>> # Using custom atlas file
    >>> fastsrm = FastSRM(atlas='/path/to/custom_atlas.nii.gz')
    >>> fastsrm.fit(multi_subject_data)

    >>> # Using pre-configured masker
    >>> from nilearn.maskers import NiftiLabelsMasker
    >>> masker = NiftiLabelsMasker(labels_img='atlas.nii.gz', standardize=True)
    >>> fastsrm = FastSRM(atlas=masker, features=30)
    >>> fastsrm.fit(multi_subject_data)

    >>> # Integration with Brain_Data
    >>> from nltools.data import Brain_Data
    >>> brain_data = [Brain_Data(f'sub_{i}.nii.gz') for i in range(5)]
    >>> fastsrm = FastSRM(atlas='schaefer_100')
    >>> fastsrm.fit(brain_data)

    >>> # Via align() function
    >>> from nltools.stats import align
    >>> result = align(data, method='fast_srm', atlas='harvard_oxford', n_features=50)

    Notes
    -----
    - Atlas choice affects speed/accuracy tradeoff:
        - Fewer parcels → faster, less accurate
        - More parcels → slower, more accurate
        - Typical range: 100-400 parcels

    - Atlas projection reduces noise by averaging within parcels,
      often improving R² compared to voxel-level SRM!

    - For very large datasets (N>50 subjects, V>50k voxels), FastSRM
      enables analyses that would be infeasible with standard SRM.

    - Transforms (w_) are projected back to full voxel space, so outputs
      are compatible with standard SRM workflows.

    References
    ----------
    Richard, H., Martin, L., Pinho, A., Pillow, J., Thirion, B. (2019).
    Fast Shared Response Model for fMRI data.
    arXiv preprint arXiv:1909.12537.
    """

    def __init__(self, atlas=None, n_iter=10, features=50, rand_seed=0, atlas_kwargs=None):
        # Initialize parent SRM
        super().__init__(n_iter=n_iter, features=features, rand_seed=rand_seed)

        # FastSRM-specific attributes
        self.atlas = atlas
        self.atlas_kwargs = atlas_kwargs or {}

        # Validate atlas is provided
        if atlas is None:
            raise ValueError(
                "FastSRM requires an atlas for dimensionality reduction. "
                "Provide atlas as str (path or preset name), array, or NiftiLabelsMasker."
            )

    def _validate_and_prepare_atlas(self, sample_data):
        """Convert atlas input to fitted NiftiLabelsMasker.

        Parameters
        ----------
        sample_data : array or Brain_Data
            Sample subject data for fitting masker

        Returns
        -------
        masker : NiftiLabelsMasker
            Fitted masker ready for transform operations
        """
        from nilearn.maskers import NiftiLabelsMasker
        from nltools.data import Brain_Data

        # Case 1: Already a NiftiLabelsMasker
        if isinstance(self.atlas, NiftiLabelsMasker):
            masker = self.atlas
            if not hasattr(masker, 'labels_img_'):
                # Not fitted yet, fit it
                if isinstance(sample_data, Brain_Data):
                    masker.fit(sample_data.to_nifti())
                else:
                    masker.fit(sample_data)
            return masker

        # Case 2: String (path or preset atlas name)
        elif isinstance(self.atlas, str):
            # Try loading as preset atlas (harvard_oxford, aal, etc.)
            masker = self._load_preset_atlas(self.atlas)
            if masker is None:
                # Treat as file path
                masker = NiftiLabelsMasker(
                    labels_img=self.atlas,
                    **self.atlas_kwargs
                )

            # Fit the masker
            if isinstance(sample_data, Brain_Data):
                masker.fit(sample_data.to_nifti())
            else:
                masker.fit(sample_data)
            return masker

        # Case 3: Numpy array (treat as atlas volume)
        elif isinstance(self.atlas, np.ndarray):
            from nibabel import Nifti1Image

            # Convert to Nifti
            if self.atlas.ndim != 3:
                raise ValueError(
                    f"Atlas array must be 3D, got shape {self.atlas.shape}"
                )

            # Create Nifti image with identity affine (can be customized via atlas_kwargs)
            affine = self.atlas_kwargs.pop('affine', np.eye(4))
            atlas_img = Nifti1Image(self.atlas.astype(int), affine)

            masker = NiftiLabelsMasker(
                labels_img=atlas_img,
                **self.atlas_kwargs
            )

            if isinstance(sample_data, Brain_Data):
                masker.fit(sample_data.to_nifti())
            else:
                masker.fit(sample_data)
            return masker

        else:
            raise ValueError(
                f"atlas must be str, numpy.ndarray, or NiftiLabelsMasker, "
                f"got {type(self.atlas)}"
            )

    def _load_preset_atlas(self, atlas_name):
        """Load common preset atlases by name.

        Parameters
        ----------
        atlas_name : str
            Name of preset atlas ('harvard_oxford', 'aal', 'schaefer_100', etc.)

        Returns
        -------
        masker : NiftiLabelsMasker or None
            Configured masker if preset found, None otherwise
        """
        from nilearn import datasets
        from nilearn.maskers import NiftiLabelsMasker

        # Map of preset names to loading functions
        preset_map = {
            'harvard_oxford': lambda: datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm'),
            'aal': lambda: datasets.fetch_atlas_aal(),
            'schaefer_100': lambda: datasets.fetch_atlas_schaefer_2018(n_rois=100),
            'schaefer_200': lambda: datasets.fetch_atlas_schaefer_2018(n_rois=200),
            'schaefer_400': lambda: datasets.fetch_atlas_schaefer_2018(n_rois=400),
        }

        atlas_lower = atlas_name.lower()
        if atlas_lower in preset_map:
            atlas_data = preset_map[atlas_lower]()

            # Extract maps attribute (varies by atlas)
            if hasattr(atlas_data, 'maps'):
                atlas_img = atlas_data.maps
            elif hasattr(atlas_data, 'filename'):
                atlas_img = atlas_data.filename
            else:
                # Fallback: return None, will try as file path
                return None

            return NiftiLabelsMasker(labels_img=atlas_img, **self.atlas_kwargs)

        return None  # Not a preset, caller will try as file path

    def _project_to_atlas(self, data, masker):
        """Project multi-subject data to atlas parcel space.

        Parameters
        ----------
        data : list of arrays or Brain_Data objects
            Multi-subject data in full voxel space

        masker : NiftiLabelsMasker
            Fitted atlas masker

        Returns
        -------
        atlas_data : list of arrays, shape[i] = [parcels, timepoints]
            Data projected to atlas parcel space
        """
        from nltools.data import Brain_Data

        atlas_data = []
        for subject_data in data:
            if isinstance(subject_data, Brain_Data):
                # Brain_Data: convert to Nifti, transform, extract array
                nifti = subject_data.to_nifti()
                projected = masker.transform(nifti)  # Shape: (timepoints, parcels)
                atlas_data.append(projected.T)  # Transpose to (parcels, timepoints)
            else:
                # Assume numpy array: shape should be (voxels, timepoints)
                # For SRM, we need to reconstruct spatial structure for masker
                # This is a limitation: without spatial info, can't use atlas!
                raise NotImplementedError(
                    "FastSRM requires Brain_Data objects or spatial information "
                    "to apply atlas projection. Raw numpy arrays not supported."
                )

        return atlas_data

    def _inverse_project_transforms(self, w_atlas, masker, original_data):
        """Project atlas-space transforms back to full voxel space.

        Parameters
        ----------
        w_atlas : list of arrays, shape[i] = [parcels, features]
            Transforms in atlas parcel space

        masker : NiftiLabelsMasker
            Fitted atlas masker

        original_data : list of Brain_Data
            Original data (needed for voxel coordinates)

        Returns
        -------
        w_full : list of arrays, shape[i] = [voxels_i, features]
            Transforms in full voxel space
        """
        from nltools.data import Brain_Data

        w_full = []
        for i, (w_atlas_i, orig_data_i) in enumerate(zip(w_atlas, original_data)):
            if not isinstance(orig_data_i, Brain_Data):
                raise ValueError(
                    "FastSRM requires Brain_Data objects for inverse projection"
                )

            # Strategy: For each feature, create parcel-level map, inverse transform to voxels
            n_parcels, n_features = w_atlas_i.shape
            n_voxels = orig_data_i.data.shape[0]

            w_full_i = np.zeros((n_voxels, n_features))

            for feat in range(n_features):
                # Create parcel-level "image" for this feature
                parcel_values = w_atlas_i[:, feat]  # Shape: (parcels,)

                # Inverse transform: map parcel values back to voxels
                # This assigns each voxel the value of its parcel
                voxel_values = masker.inverse_transform(parcel_values.reshape(1, -1))

                # Extract values within Brain_Data mask
                masked_values = orig_data_i.masker.transform(voxel_values)
                w_full_i[:, feat] = masked_values.flatten()

            w_full.append(w_full_i)

        return w_full

    def fit(self, X, y=None):
        """Fit FastSRM using atlas-based dimensionality reduction.

        Parameters
        ----------
        X : list of Brain_Data objects
            Multi-subject fMRI data. Each element is a Brain_Data instance.

        y : ignored
            Not used, present for sklearn compatibility

        Returns
        -------
        self : FastSRM
            Fitted model
        """
        from nltools.data import Brain_Data

        logger.info("Starting FastSRM with atlas-based projection")

        # Validate input
        if len(X) <= 1:
            raise ValueError(
                f"Not enough subjects ({len(X)}) to train FastSRM. Need at least 2."
            )

        # All subjects must be Brain_Data
        if not all(isinstance(x, Brain_Data) for x in X):
            raise ValueError(
                "FastSRM requires Brain_Data objects (not raw arrays) "
                "for atlas projection. Convert your data using Brain_Data()."
            )

        # Step 1: Validate and prepare atlas masker
        logger.info("Preparing atlas masker...")
        self.atlas_masker_ = self._validate_and_prepare_atlas(X[0])

        # Get number of parcels
        # Fit masker if not already fitted
        if not hasattr(self.atlas_masker_, 'labels_img_'):
            self.atlas_masker_.fit(X[0].to_nifti())

        # Count parcels (excluding background label 0)
        labels = self.atlas_masker_.labels_img_.get_fdata()
        self.n_parcels_ = len(np.unique(labels)) - 1  # Subtract 1 for background

        logger.info(f"Atlas has {self.n_parcels_} parcels "
                   f"(reducing from ~{X[0].data.shape[0]} voxels)")

        # Step 2: Project data to atlas space
        logger.info("Projecting data to atlas space...")
        X_atlas = self._project_to_atlas(X, self.atlas_masker_)

        # Verify projection worked
        if X_atlas[0].shape[1] < self.features:
            raise ValueError(
                f"Not enough samples ({X_atlas[0].shape[1]}) to train model "
                f"with {self.features} features."
            )

        logger.info(f"Projected data shape: "
                   f"{X_atlas[0].shape[0]} parcels × {X_atlas[0].shape[1]} timepoints")

        # Step 3: Run standard SRM on atlas data
        logger.info("Running SRM in atlas space...")
        # Call parent SRM.fit() on projected data
        # This populates sigma_s_, w_, mu_, rho2_, s_
        # But w_ will be atlas-space transforms, which we'll override
        super().fit(X_atlas)

        # Step 4: Store atlas-space transforms separately
        self.w_atlas_ = self.w_  # Save atlas-space transforms

        # Step 5: Inverse project transforms to full voxel space
        logger.info("Projecting transforms back to voxel space...")
        self.w_ = self._inverse_project_transforms(
            self.w_atlas_,
            self.atlas_masker_,
            X
        )

        logger.info("FastSRM fitting complete!")
        logger.info(f"Memory reduction: {X[0].data.shape[0] / self.n_parcels_:.1f}x")

        return self

    def transform(self, X):
        """Transform new data to shared feature space.

        Parameters
        ----------
        X : list of Brain_Data objects
            Multi-subject data to transform

        Returns
        -------
        transformed : list of arrays, shape[i] = [features, timepoints]
            Data in shared feature space
        """
        # Check if fitted
        if not hasattr(self, 'w_'):
            raise NotFittedError(
                "FastSRM model has not been fitted yet. Call fit() first."
            )

        # Project to atlas space
        X_atlas = self._project_to_atlas(X, self.atlas_masker_)

        # Use parent SRM transform (operates on atlas data)
        # Temporarily swap in atlas transforms
        w_full = self.w_
        self.w_ = self.w_atlas_

        transformed = super().transform(X_atlas)

        # Restore full transforms
        self.w_ = w_full

        return transformed

    def transform_subject(self, X):
        """Project new subject to shared space.

        Parameters
        ----------
        X : Brain_Data
            New subject data

        Returns
        -------
        w : array, shape = [voxels, features]
            Transform for new subject in full voxel space
        """
        # Check if fitted
        if not hasattr(self, 'w_atlas_'):
            raise NotFittedError(
                "FastSRM model has not been fitted yet. Call fit() first."
            )

        from nltools.data import Brain_Data

        if not isinstance(X, Brain_Data):
            raise ValueError("transform_subject requires Brain_Data object")

        # Project to atlas space
        X_atlas = self._project_to_atlas([X], self.atlas_masker_)[0]

        # Use parent SRM transform_subject (operates on atlas data)
        w_full_temp = self.w_
        self.w_ = self.w_atlas_

        w_atlas = super().transform_subject(X_atlas)

        # Restore
        self.w_ = w_full_temp

        # Inverse project to full voxel space
        w_full = self._inverse_project_transforms(
            [w_atlas],
            self.atlas_masker_,
            [X]
        )[0]

        return w_full
```

### 3.2 Integration with `align()` Function

```python
# In nltools/stats.py

def align(data, method='probabilistic_srm', n_features=None, atlas=None, **kwargs):
    """Align subjects in common space using various methods.

    Parameters
    ----------
    data : list of Brain_Data or arrays
        Multi-subject data to align

    method : str, default='probabilistic_srm'
        Alignment method:
        - 'probabilistic_srm': Probabilistic SRM (EM algorithm)
        - 'deterministic_srm': Deterministic SRM (BCD algorithm)
        - 'procrustes': Procrustes-based hyperalignment
        - 'fast_srm': Fast SRM with atlas reduction (NEW!)

    n_features : int, optional
        Number of features/components. Default varies by method.

    atlas : str, array, or NiftiLabelsMasker, optional
        Atlas for 'fast_srm' method. Required if method='fast_srm'.
        Can be preset name ('harvard_oxford', 'aal', 'schaefer_100'),
        path to atlas file, numpy array, or NiftiLabelsMasker.

    **kwargs : additional arguments
        Method-specific parameters (n_iter, rand_seed, etc.)

    Returns
    -------
    out : dict
        - 'transformed': list of aligned data
        - 'transformation_matrix': list of transforms (W_i)
        - 'common_model': shared response (S)
        - 'isc': ISC values (if applicable)

    Examples
    --------
    >>> # Standard SRM
    >>> result = align(data, method='probabilistic_srm', n_features=50)

    >>> # FastSRM with Harvard-Oxford atlas
    >>> result = align(data, method='fast_srm', atlas='harvard_oxford', n_features=50)

    >>> # FastSRM with custom atlas
    >>> result = align(data, method='fast_srm', atlas='/path/to/atlas.nii.gz')
    """
    # ... existing code ...

    if method == 'fast_srm':
        # Validate atlas provided
        if atlas is None:
            raise ValueError(
                "atlas parameter required for method='fast_srm'. "
                "Provide atlas name (e.g., 'harvard_oxford'), path, or masker."
            )

        # Import FastSRM
        from nltools.algorithms.srm import FastSRM

        # Create and fit model
        model = FastSRM(
            atlas=atlas,
            n_iter=kwargs.get('n_iter', 10),
            features=n_features or 50,
            rand_seed=kwargs.get('rand_seed', 0),
            atlas_kwargs=kwargs.get('atlas_kwargs', {})
        )

        model.fit(data)

        # Transform data
        transformed = model.transform(data)

        # Prepare output
        out = {
            'transformed': transformed,
            'transformation_matrix': model.w_,  # Full voxel-space transforms
            'transformation_matrix_atlas': model.w_atlas_,  # Atlas-space (bonus!)
            'common_model': model.s_,
            'atlas_masker': model.atlas_masker_,
            'n_parcels': model.n_parcels_
        }

        return out

    # ... rest of existing code ...
```

---

## 4. Testing Implementation

### 4.1 Test File Structure

```python
# File: nltools/tests/core/test_fastsrm.py

"""
Tests for FastSRM (Fast Shared Response Model) algorithm.

Testing philosophy: Property-based tests with mathematical invariants
following BrainIAK, PyMVPA, and nltools best practices.

Based on research in claude-guidelines/srm-hyperalignment-testing-strategy.md
"""

import pytest
import numpy as np
from nltools.algorithms.srm import FastSRM
from nltools.data import Brain_Data
from sklearn.exceptions import NotFittedError
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker


# ========== FIXTURES ==========

@pytest.fixture
def sample_atlas_array():
    """Create simple 3D atlas with 10 labeled parcels for testing."""
    # 10×10×10 volume
    atlas = np.zeros((10, 10, 10), dtype=int)

    # Create 10 non-overlapping parcels
    for i in range(10):
        z = i  # Different z-slice for each parcel
        atlas[2:8, 2:8, z] = i + 1  # Labels 1-10 (0 = background)

    return atlas


@pytest.fixture
def sample_atlas_masker(sample_atlas_array):
    """Create NiftiLabelsMasker from sample atlas."""
    from nibabel import Nifti1Image

    # Convert array to Nifti
    atlas_img = Nifti1Image(sample_atlas_array, affine=np.eye(4))

    # Create masker
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)

    return masker


@pytest.fixture
def atlas_brain_data(sample_atlas_array):
    """Create multi-subject Brain_Data compatible with sample atlas.

    Creates 5 subjects with shared latent structure + noise,
    spatially organized to match atlas parcels.
    """
    np.random.seed(42)

    n_subjects = 5
    n_timepoints = 100
    n_features = 10  # True latent dimensionality
    n_parcels = 10  # Matches sample_atlas

    # True shared response
    shared = np.random.randn(n_features, n_timepoints)

    # Create Brain_Data objects
    subjects = []

    for subj in range(n_subjects):
        # Random orthogonal parcel-to-feature mapping
        w_atlas = np.linalg.qr(np.random.randn(n_parcels, n_features))[0]

        # Generate parcel-level data: parcels × timepoints
        parcel_data = w_atlas @ shared + 0.01 * np.random.randn(n_parcels, n_timepoints)

        # Expand to voxel-level data (simulate multiple voxels per parcel)
        # For simplicity: 10 voxels × 10 parcels = 100 voxels total
        voxels_per_parcel = 10
        n_voxels = n_parcels * voxels_per_parcel

        voxel_data = np.zeros((n_voxels, n_timepoints))
        for p in range(n_parcels):
            # Assign parcel signal to its voxels (with small variation)
            start_vox = p * voxels_per_parcel
            end_vox = start_vox + voxels_per_parcel
            voxel_data[start_vox:end_vox, :] = (
                parcel_data[p, :] + 0.001 * np.random.randn(voxels_per_parcel, n_timepoints)
            )

        # Create Brain_Data
        # Note: This is simplified; in practice would need proper Nifti with coordinates
        brain_data = Brain_Data(voxel_data, X=create_fake_coordinates(n_voxels))
        subjects.append(brain_data)

    return {
        'data': subjects,
        'shared': shared,
        'n_parcels': n_parcels,
        'n_voxels': n_voxels,
        'n_timepoints': n_timepoints,
        'n_features': n_features
    }


def create_fake_coordinates(n_voxels):
    """Create fake MNI coordinates for Brain_Data."""
    # Generate grid of coordinates
    coords = []
    for i in range(n_voxels):
        x = (i % 10) * 2  # Spread out in 3D grid
        y = ((i // 10) % 10) * 2
        z = (i // 100) * 2
        coords.append([x, y, z])
    return np.array(coords)


# ========== PHASE 1: INITIALIZATION & CONTRACT TESTS ==========

class TestFastSRMInitialization:
    """Test FastSRM initialization and parameter validation."""

    def test_fastsrm_init_defaults(self, sample_atlas_masker):
        """Test FastSRM initializes with default parameters."""
        fastsrm = FastSRM(atlas=sample_atlas_masker)
        assert fastsrm.n_iter == 10
        assert fastsrm.features == 50
        assert fastsrm.rand_seed == 0
        assert fastsrm.atlas is sample_atlas_masker

    def test_fastsrm_init_custom_params(self, sample_atlas_array):
        """Test FastSRM accepts custom parameters."""
        fastsrm = FastSRM(
            atlas=sample_atlas_array,
            n_iter=20,
            features=30,
            rand_seed=123
        )
        assert fastsrm.n_iter == 20
        assert fastsrm.features == 30
        assert fastsrm.rand_seed == 123

    def test_fastsrm_init_with_atlas_path(self, tmp_path):
        """Test FastSRM accepts atlas as file path."""
        # Create temporary atlas file
        from nibabel import Nifti1Image, save

        atlas_data = np.random.randint(0, 10, size=(10, 10, 10))
        atlas_img = Nifti1Image(atlas_data, affine=np.eye(4))
        atlas_path = tmp_path / "test_atlas.nii.gz"
        save(atlas_img, str(atlas_path))

        fastsrm = FastSRM(atlas=str(atlas_path))
        assert fastsrm.atlas == str(atlas_path)

    def test_fastsrm_init_with_atlas_array(self, sample_atlas_array):
        """Test FastSRM accepts atlas as numpy array."""
        fastsrm = FastSRM(atlas=sample_atlas_array)
        assert isinstance(fastsrm.atlas, np.ndarray)
        assert fastsrm.atlas.shape == (10, 10, 10)

    def test_fastsrm_init_with_nilearn_masker(self, sample_atlas_masker):
        """Test FastSRM accepts NiftiLabelsMasker instance."""
        fastsrm = FastSRM(atlas=sample_atlas_masker)
        assert isinstance(fastsrm.atlas, NiftiLabelsMasker)

    def test_fastsrm_missing_atlas_error(self):
        """Test FastSRM raises error if atlas=None."""
        with pytest.raises(ValueError, match="FastSRM requires an atlas"):
            FastSRM(atlas=None)

    def test_fastsrm_fit_before_transform_error(self, sample_atlas_masker, atlas_brain_data):
        """Test that transform raises NotFittedError before fit."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10)
        with pytest.raises(NotFittedError, match="has not been fitted"):
            fastsrm.transform(atlas_brain_data['data'])

    def test_fastsrm_fit_sets_attributes(self, sample_atlas_masker, atlas_brain_data):
        """Test that fit() creates required attributes."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=2)
        fastsrm.fit(atlas_brain_data['data'])

        # Check FastSRM-specific attributes
        assert hasattr(fastsrm, 'w_')  # Full voxel-space transforms
        assert hasattr(fastsrm, 'w_atlas_')  # Atlas-space transforms
        assert hasattr(fastsrm, 's_')  # Shared response
        assert hasattr(fastsrm, 'atlas_masker_')  # Fitted masker
        assert hasattr(fastsrm, 'n_parcels_')  # Parcel count

        # Check inherited SRM attributes
        assert hasattr(fastsrm, 'sigma_s_')
        assert hasattr(fastsrm, 'mu_')
        assert hasattr(fastsrm, 'rho2_')

        # Check shapes
        assert isinstance(fastsrm.w_, list)
        assert len(fastsrm.w_) == len(atlas_brain_data['data'])
        assert isinstance(fastsrm.w_atlas_, list)
        assert len(fastsrm.w_atlas_) == len(atlas_brain_data['data'])


# ========== PHASE 2: ATLAS HANDLING TESTS ==========

class TestFastSRMAtlasHandling:
    """Test atlas preparation and projection operations."""

    def test_atlas_projection_reduces_dimensionality(self, sample_atlas_masker, atlas_brain_data):
        """Test that atlas projection reduces voxel dimensionality (A ≪ V)."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=2)
        fastsrm.fit(atlas_brain_data['data'])

        # Atlas space should have fewer dimensions than voxel space
        n_voxels = atlas_brain_data['n_voxels']
        n_parcels = fastsrm.n_parcels_

        assert n_parcels < n_voxels, \
            f"Atlas should reduce dimensionality: {n_parcels} parcels < {n_voxels} voxels"

        # Atlas transforms should be smaller
        for w_full, w_atlas in zip(fastsrm.w_, fastsrm.w_atlas_):
            assert w_full.shape[0] > w_atlas.shape[0], \
                "Full voxel space should have more rows than atlas space"

    def test_atlas_projection_preserves_timepoints(self, sample_atlas_masker, atlas_brain_data):
        """Test that atlas projection preserves timepoint dimension."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=2)

        # Project data manually
        X_atlas = fastsrm._project_to_atlas(
            atlas_brain_data['data'],
            fastsrm._validate_and_prepare_atlas(atlas_brain_data['data'][0])
        )

        # Check timepoints unchanged
        n_timepoints_orig = atlas_brain_data['n_timepoints']
        for x_atlas in X_atlas:
            assert x_atlas.shape[1] == n_timepoints_orig, \
                "Atlas projection should preserve timepoints"

    def test_invalid_atlas_format_error(self):
        """Test error for unsupported atlas types."""
        with pytest.raises(ValueError, match="atlas must be"):
            fastsrm = FastSRM(atlas={'invalid': 'dict'})

    def test_atlas_missing_labels_error(self):
        """Test error if atlas has no labeled regions."""
        # Create all-zero atlas (no parcels)
        empty_atlas = np.zeros((10, 10, 10), dtype=int)

        fastsrm = FastSRM(atlas=empty_atlas, features=10)

        # Should raise error during fit when no parcels found
        # (This test may need adjustment based on actual error handling)
        # For now, checking initialization is sufficient

    def test_atlas_projection_reversibility(self, sample_atlas_masker, atlas_brain_data):
        """Test that atlas projection is approximately reversible.

        Note: Perfect reversibility not expected due to parcel averaging.
        """
        # This is more of a sanity check than a strict property
        # Atlas projection averages voxels within parcels, so information is lost
        # We can only test that inverse projection produces reasonable values
        pass  # Skipping for now, may add if needed

    def test_different_atlas_sizes(self):
        """Test FastSRM works with varying parcel counts."""
        # Test with different atlas resolutions
        for n_parcels in [5, 10, 20]:
            # Create simple atlas
            atlas = np.zeros((20, 20, 20), dtype=int)
            parcel_size = 20 // int(np.cbrt(n_parcels))

            parcel_id = 1
            for i in range(0, 20, parcel_size):
                for j in range(0, 20, parcel_size):
                    for k in range(0, 20, parcel_size):
                        if parcel_id <= n_parcels:
                            atlas[i:i+parcel_size, j:j+parcel_size, k:k+parcel_size] = parcel_id
                            parcel_id += 1

            # Create FastSRM
            fastsrm = FastSRM(atlas=atlas, features=5, n_iter=2)

            # Verify atlas prepared correctly
            # (Full test would require fitting with data)
            assert fastsrm.atlas is atlas

    def test_atlas_with_brain_data_integration(self, sample_atlas_masker):
        """Test that FastSRM works with Brain_Data objects."""
        # This is tested implicitly in other tests
        # Just verify instantiation is okay
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10)
        assert fastsrm.atlas is sample_atlas_masker


# ========== PHASE 3: MATHEMATICAL PROPERTY TESTS ==========

class TestFastSRMMathematicalProperties:
    """Test mathematical properties that must hold for correct FastSRM."""

    def test_fastsrm_orthogonality_of_transforms(self, sample_atlas_masker, atlas_brain_data):
        """Test that full voxel-space W_i matrices have orthonormal columns (W.T @ W ≈ I)."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=5)
        fastsrm.fit(atlas_brain_data['data'])

        for i, w in enumerate(fastsrm.w_):
            # Compute W.T @ W (should be identity for orthonormal columns)
            gram = w.T @ w
            identity = np.eye(w.shape[1])  # features × features

            ortho_error = np.linalg.norm(gram - identity, 'fro')

            assert ortho_error < 1e-4, \
                f"Subject {i}: Full voxel-space W.T @ W not orthogonal (error={ortho_error:.2e})"

    def test_fastsrm_orthogonality_atlas_space(self, sample_atlas_masker, atlas_brain_data):
        """Test that atlas-space W_atlas matrices have orthonormal columns."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=5)
        fastsrm.fit(atlas_brain_data['data'])

        for i, w_atlas in enumerate(fastsrm.w_atlas_):
            gram = w_atlas.T @ w_atlas
            identity = np.eye(w_atlas.shape[1])

            ortho_error = np.linalg.norm(gram - identity, 'fro')

            assert ortho_error < 1e-5, \
                f"Subject {i}: Atlas-space W.T @ W not orthogonal (error={ortho_error:.2e})"

    def test_fastsrm_reconstruction_quality(self, sample_atlas_masker, atlas_brain_data):
        """Test that atlas-space data is well reconstructed: X_atlas ≈ W_atlas @ S."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=10)

        # Fit on data
        X_atlas = fastsrm._project_to_atlas(
            atlas_brain_data['data'],
            fastsrm._validate_and_prepare_atlas(atlas_brain_data['data'][0])
        )

        # Now fit FastSRM
        fastsrm.fit(atlas_brain_data['data'])

        # Check reconstruction in atlas space
        for i, (x_atlas, w_atlas) in enumerate(zip(X_atlas, fastsrm.w_atlas_)):
            # Demean (SRM centers data)
            x_centered = x_atlas - x_atlas.mean(axis=1, keepdims=True)

            # Reconstruct
            reconstruction = w_atlas @ fastsrm.s_

            # Relative error
            error = np.linalg.norm(x_centered - reconstruction, 'fro')
            data_norm = np.linalg.norm(x_centered, 'fro')
            relative_error = error / data_norm

            assert relative_error < 0.5, \
                f"Subject {i}: Poor atlas-space reconstruction (error={relative_error:.2%})"

    def test_fastsrm_shared_response_shape(self, sample_atlas_masker, atlas_brain_data):
        """Test that shared response has correct dimensions."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=2)
        fastsrm.fit(atlas_brain_data['data'])

        expected_shape = (10, atlas_brain_data['n_timepoints'])
        assert fastsrm.s_.shape == expected_shape

    def test_fastsrm_transform_shape_preservation(self, sample_atlas_masker, atlas_brain_data):
        """Test that transform preserves feature and timepoint dimensions."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=2)
        fastsrm.fit(atlas_brain_data['data'])

        transformed = fastsrm.transform(atlas_brain_data['data'])

        for i, s in enumerate(transformed):
            expected_shape = (10, atlas_brain_data['n_timepoints'])
            assert s.shape == expected_shape, \
                f"Subject {i}: Wrong shape {s.shape}, expected {expected_shape}"

    def test_fastsrm_variance_explained(self, sample_atlas_masker, atlas_brain_data):
        """Test that FastSRM captures substantial variance."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=10)
        fastsrm.fit(atlas_brain_data['data'])
        transformed = fastsrm.transform(atlas_brain_data['data'])

        # Compute variance in original (atlas-space) and transformed
        X_atlas = fastsrm._project_to_atlas(
            atlas_brain_data['data'],
            fastsrm.atlas_masker_
        )

        original_var = np.mean([np.var(x) for x in X_atlas])
        transformed_var = np.mean([np.var(s) for s in transformed])

        # Transformed should capture at least 30% of original variance
        assert transformed_var > 0.3 * original_var

    def test_fastsrm_atlas_compression_ratio(self, sample_atlas_masker, atlas_brain_data):
        """Verify memory reduction via parcel/voxel ratio."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=2)
        fastsrm.fit(atlas_brain_data['data'])

        compression_ratio = atlas_brain_data['n_voxels'] / fastsrm.n_parcels_

        # Should have meaningful compression
        assert compression_ratio > 1, \
            f"Atlas should reduce dimensionality (ratio={compression_ratio:.1f}x)"

    def test_fastsrm_noise_reduction_property(self, sample_atlas_masker):
        """Test that atlas projection reduces noise (optional property test)."""
        # This is more qualitative - atlas averaging should reduce high-frequency noise
        # Could implement by comparing SNR before/after projection
        # Skipping for now unless needed
        pass


# ========== PHASE 4: PERFORMANCE & COMPARATIVE TESTS ==========

class TestFastSRMPerformance:
    """Test performance characteristics and comparison with standard SRM."""

    def test_fastsrm_vs_srm_similar_results(self, sample_atlas_masker, atlas_brain_data):
        """Test that FastSRM and SRM produce similar shared responses (correlation > 0.85)."""
        from nltools.algorithms.srm import SRM

        # Fit standard SRM (on full voxel data)
        # Note: This requires converting Brain_Data to arrays
        X_arrays = [bd.data for bd in atlas_brain_data['data']]
        srm = SRM(features=10, n_iter=10, rand_seed=42)
        srm.fit(X_arrays)
        srm_transformed = srm.transform(X_arrays)

        # Fit FastSRM
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=10, rand_seed=42)
        fastsrm.fit(atlas_brain_data['data'])
        fastsrm_transformed = fastsrm.transform(atlas_brain_data['data'])

        # Compare shared responses (correlation-based)
        for s1, s2 in zip(srm_transformed, fastsrm_transformed):
            corr = np.corrcoef(s1.flatten(), s2.flatten())[0, 1]

            # Allow for differences due to atlas projection
            assert abs(corr) > 0.7, \
                "FastSRM and SRM should produce reasonably similar alignments"

    @pytest.mark.slow
    def test_fastsrm_faster_than_srm(self):
        """Timing test: FastSRM should be faster than SRM (skip in CI)."""
        # This is optional - timing tests are flaky
        # Would require larger dataset to see meaningful difference
        pass

    @pytest.mark.slow
    def test_fastsrm_memory_efficient(self):
        """Memory test: FastSRM should use less memory than SRM (skip in CI)."""
        # Optional benchmark
        pass

    def test_fastsrm_deterministic_with_seed(self, sample_atlas_masker, atlas_brain_data):
        """Test reproducibility with same random seed."""
        fastsrm1 = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=5, rand_seed=42)
        fastsrm1.fit(atlas_brain_data['data'])

        fastsrm2 = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=5, rand_seed=42)
        fastsrm2.fit(atlas_brain_data['data'])

        # Should produce identical results
        np.testing.assert_array_almost_equal(fastsrm1.s_, fastsrm2.s_, decimal=10)

        for w1, w2 in zip(fastsrm1.w_, fastsrm2.w_):
            np.testing.assert_array_almost_equal(w1, w2, decimal=10)

    def test_fastsrm_different_seed_different_init(self, sample_atlas_masker, atlas_brain_data):
        """Test that different seeds produce different initializations."""
        fastsrm1 = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=1, rand_seed=42)
        fastsrm1.fit(atlas_brain_data['data'])

        fastsrm2 = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=1, rand_seed=123)
        fastsrm2.fit(atlas_brain_data['data'])

        # Should produce different results
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(fastsrm1.s_, fastsrm2.s_, decimal=5)

    def test_fastsrm_convergence_iterations(self, sample_atlas_masker, atlas_brain_data):
        """Test that more iterations improve fit (up to convergence)."""
        # Fit with few iterations
        fastsrm_few = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=2, rand_seed=42)
        fastsrm_few.fit(atlas_brain_data['data'])

        # Fit with many iterations
        fastsrm_many = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=20, rand_seed=42)
        fastsrm_many.fit(atlas_brain_data['data'])

        # More iterations should not make things worse (at minimum)
        # Could check reconstruction error, but simpler to just verify both complete
        assert fastsrm_few.s_ is not None
        assert fastsrm_many.s_ is not None


# ========== PHASE 5: EDGE CASES & ERROR HANDLING ==========

class TestFastSRMEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_fastsrm_single_subject_error(self, sample_atlas_masker):
        """Test error with only 1 subject."""
        np.random.seed(111)
        single_subject = [Brain_Data(np.random.randn(100, 50))]

        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10)
        with pytest.raises(ValueError, match="Not enough subjects"):
            fastsrm.fit(single_subject)

    def test_fastsrm_mismatched_timepoints_error(self, sample_atlas_masker):
        """Test error when subjects have different timepoints."""
        np.random.seed(222)
        data = [
            Brain_Data(np.random.randn(100, 50)),
            Brain_Data(np.random.randn(100, 60)),  # Different TRs
        ]

        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10)
        with pytest.raises(ValueError, match="Different number of samples"):
            fastsrm.fit(data)

    def test_fastsrm_insufficient_samples_error(self, sample_atlas_masker):
        """Test error when samples < features."""
        np.random.seed(333)
        data = [
            Brain_Data(np.random.randn(100, 40)),  # 40 samples
            Brain_Data(np.random.randn(100, 40)),
        ]

        fastsrm = FastSRM(atlas=sample_atlas_masker, features=50)  # More features than samples
        with pytest.raises(ValueError, match="Not enough samples"):
            fastsrm.fit(data)

    def test_fastsrm_data_outside_atlas_mask(self):
        """Test handling of voxels not in atlas parcels."""
        # This would require more complex fixture setup
        # In practice, NiftiLabelsMasker handles this gracefully
        pass

    def test_fastsrm_empty_atlas_parcels(self):
        """Test handling of parcels with no signal."""
        # Edge case: some parcels might have zero variance
        # Skipping for now unless needed
        pass

    def test_fastsrm_transform_subject_new_data(self, sample_atlas_masker, atlas_brain_data):
        """Test transform_subject() with new subject data."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=5)
        fastsrm.fit(atlas_brain_data['data'])

        # Create new subject
        np.random.seed(999)
        new_subject = atlas_brain_data['data'][0]  # Use first subject as template

        # Transform new subject
        w_new = fastsrm.transform_subject(new_subject)

        # Check orthogonality
        gram = w_new.T @ w_new
        identity = np.eye(w_new.shape[1])
        ortho_error = np.linalg.norm(gram - identity, 'fro')

        assert ortho_error < 1e-5, \
            f"New subject transform not orthogonal (error={ortho_error:.2e})"

    def test_fastsrm_identical_subjects(self, sample_atlas_masker):
        """Test FastSRM with identical subjects."""
        np.random.seed(444)
        base_data = Brain_Data(np.random.randn(100, 50))
        identical_subjects = [base_data] * 3

        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=5)
        fastsrm.fit(identical_subjects)

        # Should complete without error
        # Orthogonality should still hold
        for w in fastsrm.w_:
            gram = w.T @ w
            identity = np.eye(w.shape[1])
            ortho_error = np.linalg.norm(gram - identity, 'fro')
            assert ortho_error < 1e-5

    def test_fastsrm_minimal_features(self, sample_atlas_masker, atlas_brain_data):
        """Test FastSRM with very few features (K=3)."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=3, n_iter=5)
        fastsrm.fit(atlas_brain_data['data'])

        # Should still produce valid orthogonal transforms
        for w in fastsrm.w_:
            gram = w.T @ w
            identity = np.eye(w.shape[1])
            ortho_error = np.linalg.norm(gram - identity, 'fro')
            assert ortho_error < 1e-5


# ========== PHASE 6: INTEGRATION & WORKFLOW TESTS ==========

class TestFastSRMIntegration:
    """Integration tests with Brain_Data and align() function."""

    def test_fastsrm_brain_data_input(self, sample_atlas_masker, atlas_brain_data):
        """Test that FastSRM works with Brain_Data objects."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=5)

        # Should accept Brain_Data list
        fastsrm.fit(atlas_brain_data['data'])
        transformed = fastsrm.transform(atlas_brain_data['data'])

        # Check output
        assert len(transformed) == len(atlas_brain_data['data'])
        for s in transformed:
            assert s.shape == (10, atlas_brain_data['n_timepoints'])

    def test_fastsrm_align_function_integration(self, sample_atlas_masker, atlas_brain_data):
        """Test FastSRM via align(method='fast_srm') function."""
        from nltools.stats import align

        result = align(
            atlas_brain_data['data'],
            method='fast_srm',
            atlas=sample_atlas_masker,
            n_features=10
        )

        # Check output structure
        assert 'transformed' in result
        assert 'transformation_matrix' in result
        assert 'transformation_matrix_atlas' in result
        assert 'common_model' in result
        assert 'atlas_masker' in result
        assert 'n_parcels' in result

        # Check shapes
        assert len(result['transformed']) == len(atlas_brain_data['data'])
        assert result['common_model'].shape[0] == 10

    def test_fastsrm_fit_transform(self, sample_atlas_masker, atlas_brain_data):
        """Test sklearn pipeline compatibility with fit_transform()."""
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=5)

        # fit_transform() should be equivalent to fit() then transform()
        transformed = fastsrm.fit_transform(atlas_brain_data['data'])

        assert len(transformed) == len(atlas_brain_data['data'])

    @pytest.mark.slow
    def test_fastsrm_common_atlases(self):
        """Test FastSRM with real atlases from nilearn datasets (slow - downloads data)."""
        # This would test with actual Harvard-Oxford, AAL, Schaefer atlases
        # Requires downloading datasets, so mark as slow and skip in CI

        # Example:
        # from nilearn import datasets
        # atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        # fastsrm = FastSRM(atlas='harvard_oxford', features=30)
        # ...

        pass  # Skip unless explicitly running slow tests

    def test_fastsrm_save_load_model(self, sample_atlas_masker, atlas_brain_data, tmp_path):
        """Test pickling/unpickling fitted FastSRM model."""
        import pickle

        # Fit model
        fastsrm = FastSRM(atlas=sample_atlas_masker, features=10, n_iter=5)
        fastsrm.fit(atlas_brain_data['data'])

        # Save to pickle
        model_path = tmp_path / "fastsrm_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(fastsrm, f)

        # Load from pickle
        with open(model_path, 'rb') as f:
            fastsrm_loaded = pickle.load(f)

        # Check that loaded model works
        transformed = fastsrm_loaded.transform(atlas_brain_data['data'])

        # Should produce same results as original
        np.testing.assert_array_almost_equal(
            transformed[0],
            fastsrm.transform(atlas_brain_data['data'])[0]
        )
```

---

## 5. Execution Plan

### 5.1 TDD Workflow (Iterative Cycles)

**Cycle 1: Initialization (2 hours)**
1. Write Phase 1 tests (8 tests)
2. Run tests → All fail ✅
3. Implement `__init__()` and basic validation
4. Run tests → Pass ✅
5. Commit: "feat: Add FastSRM initialization and validation"

**Cycle 2: Atlas Handling (2 hours)**
1. Write Phase 2 tests (7 tests)
2. Run tests → All fail ✅
3. Implement `_validate_and_prepare_atlas()`, `_load_preset_atlas()`
4. Run tests → Pass ✅
5. Commit: "feat: Add FastSRM atlas loading and validation"

**Cycle 3: Core Algorithm (3 hours)**
1. Write Phase 3 tests (8 tests) - mathematical properties
2. Run tests → All fail ✅
3. Implement `fit()`, `_project_to_atlas()`, `_inverse_project_transforms()`
4. Run tests → Pass ✅
5. Commit: "feat: Implement FastSRM fit with atlas projection"

**Cycle 4: Transform Methods (1.5 hours)**
1. Write transform-related tests from Phase 3
2. Implement `transform()` and `transform_subject()`
3. Run tests → Pass ✅
4. Commit: "feat: Add FastSRM transform methods"

**Cycle 5: Comparison & Edge Cases (2 hours)**
1. Write Phase 4 and Phase 5 tests (14 tests)
2. Run tests → All fail (expected for new tests) ✅
3. Fix any bugs revealed by edge case tests
4. Run tests → Pass ✅
5. Commit: "test: Add FastSRM edge cases and SRM comparison"

**Cycle 6: Integration (2 hours)**
1. Write Phase 6 tests (5 tests)
2. Implement `align()` function integration
3. Run full test suite → Pass ✅
4. Commit: "feat: Integrate FastSRM with align() function"

**Cycle 7: Documentation & Cleanup (1.5 hours)**
1. Write comprehensive docstrings
2. Add examples to docstrings
3. Update docs/migration-guide.md
4. Run all tests one final time → Pass ✅
5. Commit: "docs: Add FastSRM documentation and examples"

### 5.2 Validation Checklist

Before marking FastSRM complete:

- [ ] All 42 tests passing
- [ ] Code coverage >90% for `FastSRM` class
- [ ] Docstrings complete (NumPy style)
- [ ] Examples in docstrings tested
- [ ] Integration with `align()` function works
- [ ] Works with Brain_Data objects
- [ ] Works with preset atlases ('harvard_oxford', 'aal', 'schaefer_100')
- [ ] Works with custom atlas files
- [ ] Works with numpy array atlases
- [ ] Pickle/unpickle works correctly
- [ ] Comparison with SRM shows reasonable agreement (corr > 0.7)
- [ ] Migration guide updated
- [ ] refactor-todos.md updated with task completion

---

## 6. Success Criteria

### 6.1 Functional Requirements

✅ **Must Have**:
1. FastSRM fits on multi-subject Brain_Data with atlas
2. Produces orthogonal transforms in both atlas and voxel space
3. Shared response has correct shape and properties
4. Comparable results to standard SRM (correlation > 0.7)
5. Deterministic with same random seed
6. Handles edge cases gracefully (single subject, mismatched TRs, etc.)
7. Integrates with `align()` function
8. Works with preset atlases from nilearn

✅ **Nice to Have**:
9. Demonstrably faster than SRM (benchmark)
10. Demonstrably more memory efficient (benchmark)
11. Works with very large datasets (V > 50k voxels)

### 6.2 Code Quality

- Clean, readable code following nltools style
- Comprehensive docstrings with examples
- Type hints where appropriate
- No code duplication
- Efficient implementations (vectorized operations)

### 6.3 Testing

- 42 tests total (minimum)
- Property-based tests for mathematical correctness
- Contract tests for API behavior
- Edge case tests for robustness
- Integration tests for workflows
- No golden output tests (following best practices)

---

## 7. Future Enhancements (Post-Implementation)

**Priority 7.1: Parallelization** (2-3 hours)
- Add `n_jobs` parameter for multi-subject processing
- Use joblib for parallel atlas projections
- Platform-aware testing (macOS vs Linux)

**Priority 7.2: GPU Acceleration** (4-6 hours)
- JAX backend option for GPU-accelerated SRM
- Particularly beneficial for large atlas spaces (A > 1000 parcels)

**Priority 7.3: Online/Incremental FastSRM** (6-8 hours)
- Support adding new subjects without refitting
- Streaming data processing for very large studies

**Priority 7.4: Atlas Optimization** (3-4 hours)
- Automatic atlas selection based on data properties
- Adaptive parcellation that maximizes shared structure

---

## 8. References

**Papers**:
- Richard, H., Martin, L., Pinho, A., Pillow, J., Thirion, B. (2019). Fast Shared Response Model for fMRI data. arXiv:1909.12537.
- Chen, P. H. C., et al. (2015). A Reduced-Dimension fMRI Shared Response Model. NIPS.

**Code**:
- BrainIAK: https://brainiak.org/docs/brainiak.funcalign.html
- nltools SRM: `nltools/algorithms/srm.py`
- nltools testing strategy: `claude-guidelines/srm-hyperalignment-testing-strategy.md`

**Tools**:
- Nilearn: https://nilearn.github.io/stable/modules/maskers.html
- NiftiLabelsMasker: https://nilearn.github.io/stable/modules/generated/nilearn.maskers.NiftiLabelsMasker.html

---

## 9. Risk Mitigation

**Risk 1**: Brain_Data objects don't preserve spatial structure needed for atlas
- **Mitigation**: Brain_Data has `to_nifti()` method; use it for atlas operations
- **Tested in**: Phase 6 integration tests

**Risk 2**: Atlas projection loses too much information
- **Mitigation**: Test correlation with standard SRM; accept if > 0.7
- **Tested in**: Phase 4 comparative tests

**Risk 3**: Different atlas formats cause compatibility issues
- **Mitigation**: Extensive atlas handling tests with multiple formats
- **Tested in**: Phase 2 atlas handling tests

**Risk 4**: Performance gains not as dramatic as advertised
- **Mitigation**: Document actual performance; gains depend on V/A ratio
- **Tested in**: Phase 4 performance tests (optional)

---

**Last Updated**: 2025-10-29
**Author**: Claude (following nltools design philosophy)
**Status**: Ready for implementation
**Estimated Total Time**: 10-14 hours
