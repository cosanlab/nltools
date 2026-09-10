"""LocalAlignment: Neighborhood-based functional alignment.

Implements the ``'searchlight'`` and ``'roi'`` spatial scales (the searchlight
and piecewise schemes of Bazeille et al. 2021). Uses center-only aggregation to
preserve orthogonality of local transforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from collections.abc import Iterator

import numpy as np
from scipy.linalg import orthogonal_procrustes

from nltools.algorithms.backends import Backend, resolve_backend
from nltools.utils import make_progress_bar, maybe_tqdm

if TYPE_CHECKING:
    import nibabel as nib

    from nltools.data.braindata.neighborhoods import SphereNeighborhoods

logger = logging.getLogger(__name__)


def _orthogonal_procrustes_backend(
    A: np.ndarray, B: np.ndarray, backend: Backend
) -> np.ndarray:
    """GPU-compatible orthogonal Procrustes using `Backend.svd`.

    Finds the orthogonal matrix R that minimizes `||A - B @ R||_F`. Equivalent
    to `scipy.linalg.orthogonal_procrustes` (used directly on the numpy backend)
    but runs the SVD through the `Backend` abstraction for GPU acceleration.

    Args:
        A (np.ndarray): Target matrix, shape (n, m).
        B (np.ndarray): Matrix to transform, shape (n, m).
        backend (Backend): Backend instance for the computation.

    Returns:
        np.ndarray: Orthogonal matrix R, shape (m, m).
    """
    if backend.name == "numpy":
        # Use scipy for numpy backend (more efficient)
        R, _ = orthogonal_procrustes(A, B)
        return R

    # GPU path: Compute SVD of B.T @ A
    # The solution is R = V @ U.T where U, s, V.T = svd(B.T @ A)
    A_device = backend.to_device(A.astype(np.float32))
    B_device = backend.to_device(B.astype(np.float32))

    # M = B.T @ A
    M = backend.matmul(B_device.T if hasattr(B_device, "T") else B_device.t(), A_device)

    # SVD: M = U @ diag(s) @ Vt
    U, _, Vt = backend.svd(M, full_matrices=False)

    # R = V @ U.T = Vt.T @ U.T
    if backend.name.startswith("torch"):
        import torch

        R = torch.matmul(
            Vt.T if hasattr(Vt, "T") else Vt.t(), U.T if hasattr(U, "T") else U.t()
        )
        return backend.to_numpy(R).astype(np.float64)
    return Vt.T @ U.T


@dataclass
class RoiNeighborhoods:
    """Neighborhoods from a brain parcellation for ROI-scale alignment.

    Unlike the searchlight scale (overlapping spheres), the ROI scale uses
    non-overlapping parcels where each voxel belongs to exactly one region
    (the piecewise scheme of Bazeille et al. 2021).

    Attributes:
        parcel_to_voxels (dict[int, np.ndarray]): Maps each parcel id to the array
            of masked voxel indices it contains.
        n_voxels (int): Total number of voxels in the mask.
        n_parcels (int): Number of parcels (excluding the background label 0).
    """

    parcel_to_voxels: dict[int, np.ndarray]
    n_voxels: int
    n_parcels: int

    def iter_neighborhoods(
        self, progress_bar: bool = False
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over all parcels.

        Args:
            progress_bar (bool): Show a tqdm progress bar. Defaults to False.

        Yields:
            tuple[int, np.ndarray]: `(parcel_id, voxel_indices)` for each parcel.
        """
        iterator = self.parcel_to_voxels.items()

        yield from maybe_tqdm(
            list(iterator), progress_bar=progress_bar, desc="ROI", unit="parcels"
        )

    def __repr__(self) -> str:
        return f"RoiNeighborhoods(n_voxels={self.n_voxels}, n_parcels={self.n_parcels})"


def _compute_roi_neighborhoods(
    roi_mask: nib.Nifti1Image,
    mask: nib.Nifti1Image,
) -> RoiNeighborhoods:
    """Compute ROI-scale (parcel) neighborhoods from a parcellation image.

    Args:
        roi_mask (nib.Nifti1Image): Image with an integer label per parcel;
            background/unlabeled voxels are 0.
        mask (nib.Nifti1Image): Brain mask defining the voxel space.

    Returns:
        RoiNeighborhoods: Parcel-to-voxel mappings.

    Raises:
        ValueError: If `roi_mask` and `mask` have different shapes.
    """
    parc_data = roi_mask.get_fdata().astype(int)
    mask_data = mask.get_fdata().astype(bool)

    if parc_data.shape != mask_data.shape:
        raise ValueError(
            f"Parcellation shape {parc_data.shape} != mask shape {mask_data.shape}"
        )

    # Get masked voxel coordinates
    mask_indices = np.where(mask_data.ravel())[0]
    n_voxels = len(mask_indices)

    # Map 3D coordinates to flat masked indices
    # For each masked voxel, get its parcel label
    parc_flat = parc_data.ravel()
    parcel_labels = parc_flat[mask_indices]

    # Build parcel → voxel mapping (excluding background label 0)
    unique_parcels = np.unique(parcel_labels)
    unique_parcels = unique_parcels[unique_parcels > 0]  # Exclude background

    parcel_to_voxels: dict[int, np.ndarray] = {}
    for parcel_id in unique_parcels:
        # Find which masked voxel indices belong to this parcel
        voxel_indices = np.where(parcel_labels == parcel_id)[0]
        if len(voxel_indices) > 0:
            parcel_to_voxels[int(parcel_id)] = voxel_indices

    return RoiNeighborhoods(
        parcel_to_voxels=parcel_to_voxels,
        n_voxels=n_voxels,
        n_parcels=len(parcel_to_voxels),
    )


def _fit_one_neighborhood(
    region_id: int,
    voxel_indices: np.ndarray,
    data: list[np.ndarray],
    method: str,
    n_iter: int,
    n_features: int | None,
    backend: Backend | None = None,
) -> tuple[int, list[np.ndarray], np.ndarray]:
    """Fit alignment for a single neighborhood (the unit of parallel work).

    Args:
        region_id (int): Id of the neighborhood (center voxel) or parcel.
        voxel_indices (np.ndarray): Voxel indices in this neighborhood.
        data (list[np.ndarray]): Full subject data arrays, each
            (n_voxels, n_samples).
        method (str): One of `'procrustes'`, `'srm'`, or `'hyperalignment'`.
        n_iter (int): Number of refinement iterations.
        n_features (int | None): Number of SRM features; None uses
            `min(n_local_voxels, n_samples)`.
        backend (Backend | None): Backend for GPU acceleration; None uses numpy.

    Returns:
        tuple[int, list[np.ndarray], np.ndarray]: `(region_id, transforms,
            template)` — the per-subject transforms and the aligned template.
    """
    n_subjects = len(data)

    # Extract local data for all subjects
    local_data = [subj[voxel_indices, :] for subj in data]
    n_local_voxels = len(voxel_indices)

    # Handle unequal sample counts by zero-padding to max length
    sample_counts = [ld.shape[1] for ld in local_data]
    max_samples = max(sample_counts)
    if not all(s == max_samples for s in sample_counts):
        local_data = [
            np.hstack([ld, np.zeros((ld.shape[0], max_samples - ld.shape[1]))])
            if ld.shape[1] < max_samples
            else ld
            for ld in local_data
        ]
    n_samples = max_samples

    # Handle degenerate cases
    if n_local_voxels < 2:
        transforms = [np.eye(n_local_voxels) for _ in range(n_subjects)]
        template = np.mean(np.array(local_data), axis=0)
        return region_id, transforms, template

    # Fit based on method
    if method == "procrustes":
        transforms, template = _fit_local_procrustes(
            local_data, n_iter=n_iter, backend=backend
        )
    elif method == "srm":
        from .srm import SRM

        feat = n_features if n_features is not None else min(n_local_voxels, n_samples)
        srm = SRM(n_iter=n_iter, n_features=feat)
        srm.fit(local_data, parallel=None)
        transforms, template = srm.w_, srm.s_
    elif method == "hyperalignment":
        from .hyperalignment import HyperAlignment

        ha = HyperAlignment(n_iter=n_iter)
        ha.fit(local_data, parallel=None)
        transforms, template = ha.w_, ha.s_
    else:
        raise ValueError(f"Unknown method: {method}")

    return region_id, transforms, template


def _fit_local_procrustes(
    data: list[np.ndarray],
    n_iter: int = 3,
    backend: Backend | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Fit multi-subject Procrustes alignment on local data.

    Iteratively refines a template by aligning all subjects and averaging.
    This is a simplified version of HyperAlignment for local neighborhoods.

    Args:
        data (list[np.ndarray]): Arrays of shape (n_local_voxels, n_samples).
        n_iter (int): Number of refinement iterations. Defaults to 3.
        backend (Backend | None): Backend for GPU acceleration; None uses
            numpy/scipy.

    Returns:
        tuple[list[np.ndarray], np.ndarray]: `(transforms, template)` — the list of
            orthogonal transforms, each (n_local_voxels, n_local_voxels), and the mean
            template in aligned space, shape (n_local_voxels, n_samples).
    """
    n_subjects = len(data)
    n_voxels, n_samples = data[0].shape

    # Use numpy backend if none provided
    if backend is None:
        backend = Backend("numpy")

    # Center and normalize each subject
    centered = []
    for x in data:
        c = x - x.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        centered.append(c)

    # Initialize template as mean of centered data
    template = np.mean(centered, axis=0)

    # Iteratively refine
    transforms = [np.eye(n_voxels) for _ in range(n_subjects)]
    for _ in range(n_iter):
        aligned = []
        for i, x in enumerate(centered):
            # Solve Procrustes: min ||template - x @ R.T||_F s.t. R orthogonal
            # scipy's orthogonal_procrustes: R, scale = argmin ||A - B @ R||_F
            R = _orthogonal_procrustes_backend(template.T, x.T, backend)
            # R transforms x.T to align with template.T, so aligned = x.T @ R
            # But we want row-wise (voxels), so: aligned = R.T @ x
            transforms[i] = R.T
            aligned.append(R.T @ x)
        # Update template
        template = np.mean(aligned, axis=0)

    return transforms, template


@dataclass
class LocalAlignment:
    """Local (neighborhood-based) functional alignment across subjects.

    Learns alignment transforms within local neighborhoods (searchlight spheres
    or parcels) and applies center-only aggregation to preserve orthogonality.

    Args:
        spatial_scale (str): Spatial scale, either 'searchlight' (overlapping
            spheres) or 'roi' (non-overlapping parcels). Defaults to 'searchlight'.
        method (str): Alignment method, one of 'procrustes', 'srm', or
            'hyperalignment'. Defaults to 'procrustes'.
        radius (float): Sphere radius in millimeters for the searchlight scale.
            Defaults to 10.0.
        roi_mask (nib.Nifti1Image | None): Parcellation image for the ROI scale.
            Required if `spatial_scale='roi'`. Defaults to None.
        n_features (int | None): Number of SRM features per neighborhood. None uses
            `min(n_local_voxels, n_samples)`; ignored by the other methods.
            Defaults to None.
        n_iter (int): Number of iterations for alignment refinement. Defaults to 3.
        aggregation (str): `'center'` writes only each sphere's center voxel
            (preserves orthogonality); `'all'` writes every voxel in the region
            and is selected automatically for `spatial_scale='roi'`. Defaults to
            `'center'`.
        parallel (str | None): Parallelization mode. None runs single-threaded numpy,
            'cpu' uses joblib CPU parallelization, and 'gpu' uses PyTorch. GPU
            acceleration applies only to `method='procrustes'`; requesting
            'gpu' with the 'srm' or 'hyperalignment' methods raises
            `NotImplementedError` (an explicit GPU request never silently runs
            on CPU). Defaults to 'cpu'.
        n_jobs (int): Number of jobs for CPU parallelization. Defaults to -1.
        progress_bar (bool): Whether to display tqdm progress bars during fit and
            transform. Defaults to False.
        n_neighborhoods_batch (int | None): Number of neighborhoods to process per
            batch on the GPU. None auto-calculates a batch size from
            `memory_budget_gb`. Defaults to None.
        memory_budget_gb (float | None): Explicit memory budget (in GB) used to
            auto-size GPU batches when `n_neighborhoods_batch` is None. None
            (default) measures the device's available memory.

    Attributes:
        transforms_ (dict[int, list[np.ndarray]]): Per-neighborhood transforms. Keys
            are center voxel indices (searchlight) or parcel ids (roi); values are
            lists of transform matrices, one per subject.
        template_ (dict[int, np.ndarray]): Per-neighborhood templates used for
            alignment.
        neighborhoods_ (SphereNeighborhoods | RoiNeighborhoods): Computed
            neighborhoods (searchlight spheres or parcels).
        n_voxels_ (int): Total number of voxels in the mask.
        mask_ (nib.Nifti1Image): Brain mask used for fitting.
        backend_ (Backend): Execution backend selected from `parallel`.

    Examples:
        ```python
        import numpy as np
        import nibabel as nib
        from nltools.algorithms.alignment import LocalAlignment

        # Synthetic multi-subject data (voxels, samples) and a matching 1000-voxel mask
        data = [np.random.randn(1000, 100) for _ in range(5)]
        mask = nib.Nifti1Image(np.ones((10, 10, 10), dtype=np.int8), np.eye(4))

        la = LocalAlignment(spatial_scale="searchlight", method="procrustes", radius=10.0)
        la.fit(data, mask)
        aligned = la.transform(data)  # list of (1000, 100) arrays
        ```

    Note:
        Based on Bazeille et al. 2021, "An empirical evaluation of functional
        alignment using inter-subject decoding". Center-only aggregation
        preserves the local orthogonality of the transforms.
    """

    # Configuration
    spatial_scale: str = "searchlight"
    method: str = "procrustes"
    radius: float = 10.0
    roi_mask: nib.Nifti1Image | None = None
    n_features: int | None = None
    n_iter: int = 3
    aggregation: str = "center"
    parallel: str | None = "cpu"
    n_jobs: int = -1
    progress_bar: bool = False

    # Batching parameters (Phase 2)
    n_neighborhoods_batch: int | None = None  # None = auto-calculate
    memory_budget_gb: float | None = None  # None = measured device budget

    # Fitted state (set by fit())
    transforms_: dict[int, list[np.ndarray]] | None = field(default=None, repr=False)
    template_: dict[int, np.ndarray] | None = field(default=None, repr=False)
    neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = field(
        default=None, repr=False
    )
    n_voxels_: int | None = field(default=None, repr=False)
    mask_: nib.Nifti1Image | None = field(default=None, repr=False)
    backend_: Backend | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.spatial_scale not in ("searchlight", "roi"):
            raise ValueError(f"Unknown spatial_scale: {self.spatial_scale}")
        if self.method not in ("procrustes", "srm", "hyperalignment"):
            raise ValueError(f"Unknown method: {self.method}")
        if self.aggregation not in ("center", "all"):
            raise ValueError(
                f"Unknown aggregation: {self.aggregation}. "
                "Supported: 'center' (searchlight), 'all' (roi)."
            )
        # Validate spatial_scale/aggregation compatibility
        if self.spatial_scale == "roi" and self.aggregation == "center":
            # Auto-switch to 'all' for roi (center-only doesn't make sense)
            self.aggregation = "all"
        if self.spatial_scale == "roi" and self.roi_mask is None:
            raise ValueError("roi_mask is required for spatial_scale='roi'")
        if self.parallel not in (None, "cpu", "gpu"):
            raise ValueError(
                f"parallel must be None, 'cpu', or 'gpu', got {self.parallel!r}"
            )
        if self.parallel == "gpu" and self.method in ("srm", "hyperalignment"):
            raise NotImplementedError(
                f"parallel='gpu' is not implemented for method={self.method!r} "
                "(only 'procrustes' has a GPU path). Use parallel='cpu'."
            )

    def _init_backend(self) -> Backend:
        """Initialize backend based on parallel setting.

        Returns:
            Backend: Backend configured for the requested execution mode.

        Raises:
            ImportError: If `parallel='gpu'` and PyTorch is not installed. An
                explicit GPU request never silently degrades to CPU; use
                `parallel='cpu'` when torch is unavailable.
            RuntimeError: If PyTorch is installed but no CUDA or MPS accelerator
                is available.
        """
        if self.parallel is None or self.parallel == "cpu":
            return Backend("numpy")
        # parallel == 'gpu' (validated in __post_init__): run-or-raise.
        backend = resolve_backend("gpu")
        logger.info(f"Using backend: {backend.name}")
        return backend

    def _auto_batch_size(
        self,
        n_neighborhoods: int,
        n_subjects: int,
        avg_region_size: int,
        n_samples: int,
    ) -> int:
        """Calculate batch size based on memory budget.

        Thin adapter over the core layer in `nltools.algorithms.backends`:
        supplies the per-neighborhood working-set estimate; the budget and
        clamp policy live in `auto_batch_size`/`device_memory_budget`.

        Args:
            n_neighborhoods (int): Total number of neighborhoods to process.
            n_subjects (int): Number of subjects.
            avg_region_size (int): Average voxels per neighborhood/parcel.
            n_samples (int): Number of time samples.

        Returns:
            int: Number of neighborhoods per batch.
        """
        from ..backends import auto_batch_size, device_memory_budget

        # Memory per neighborhood during fitting:
        # - Local data: n_subjects × avg_region_size × n_samples × 8 bytes (float64)
        # - Transforms: n_subjects × avg_region_size × avg_region_size × 8 bytes
        # - Template: avg_region_size × n_samples × 8 bytes
        bytes_per_neighborhood = (
            n_subjects * avg_region_size * n_samples * 8  # local data
            + n_subjects * avg_region_size * avg_region_size * 8  # transforms
            + avg_region_size * n_samples * 8  # template
        )

        budget_gb = device_memory_budget(
            self.backend_,
            max_gpu_memory_gb=self.memory_budget_gb,
            cap_for_batching=True,
        )
        batch_size, _ = auto_batch_size(
            n_neighborhoods, bytes_per_neighborhood, budget_gb=budget_gb
        )

        logger.debug(
            f"Auto batch size: {batch_size} neighborhoods "
            f"({bytes_per_neighborhood / 1e6:.1f} MB each, "
            f"{budget_gb:.1f} GB budget)"
        )
        return batch_size

    def _batch_neighborhoods(
        self,
        neighborhoods: SphereNeighborhoods | RoiNeighborhoods,
        n_subjects: int,
        n_samples: int,
    ) -> Iterator[list[tuple[int, np.ndarray]]]:
        """Yield neighborhoods in batches sized to the memory budget.

        Only one batch is materialized at a time; the caller processes it and
        deletes it before the next is yielded.

        Args:
            neighborhoods (SphereNeighborhoods | RoiNeighborhoods): Computed
                neighborhoods (searchlight or roi).
            n_subjects (int): Number of subjects.
            n_samples (int): Number of time samples.

        Yields:
            list[tuple[int, np.ndarray]]: One batch of `(region_id, voxel_indices)`
                pairs.
        """
        # Collect all neighborhoods
        all_neighborhoods = list(neighborhoods.iter_neighborhoods(progress_bar=False))

        # Calculate batch size
        if self.n_neighborhoods_batch is not None:
            batch_size = self.n_neighborhoods_batch
        else:
            # Estimate average region size
            if len(all_neighborhoods) > 0:
                avg_region_size = int(
                    np.mean([len(indices) for _, indices in all_neighborhoods])
                )
            else:
                avg_region_size = 1
            batch_size = self._auto_batch_size(
                max(1, len(all_neighborhoods)), n_subjects, avg_region_size, n_samples
            )

        n_total = len(all_neighborhoods)
        n_batches = (n_total + batch_size - 1) // batch_size

        logger.info(
            f"Processing {n_total} neighborhoods in {n_batches} batches "
            f"(batch_size={batch_size})"
        )

        for i in range(0, n_total, batch_size):
            batch = all_neighborhoods[i : i + batch_size]
            yield batch
            # Caller should process and then del batch for memory efficiency

    def fit(self, data: list[np.ndarray], mask: nib.Nifti1Image) -> LocalAlignment:
        """Fit local alignment on multi-subject data.

        Args:
            data (list[np.ndarray]): Subject data arrays, each of shape
                (n_voxels, n_samples). Subjects may differ in the number of
                samples; shorter subjects are zero-padded within each neighborhood.
            mask (nib.Nifti1Image): Brain mask defining the voxel space.

        Returns:
            LocalAlignment: The fitted alignment model (`self`).
        """
        from nltools.data.braindata.neighborhoods import (
            compute_searchlight_neighborhoods,
        )

        # Validate inputs
        if not isinstance(data, list) or len(data) < 2:
            raise ValueError("data must be a list of at least 2 subject arrays")

        n_subjects = len(data)
        n_voxels = data[0].shape[0]
        # Allow different sample counts - underlying methods (SRM, HyperAlignment) handle padding
        sample_counts = [subj.shape[1] for subj in data]
        n_samples = max(sample_counts)  # Use max for memory estimation

        for i, subj in enumerate(data):
            if subj.shape[0] != n_voxels:
                raise ValueError(
                    f"All subjects must have same number of voxels. "
                    f"Subject 0 has {n_voxels}, subject {i} has {subj.shape[0]}"
                )

        # Store mask and voxel count
        self.mask_ = mask
        self.n_voxels_ = n_voxels

        # Initialize backend
        self.backend_ = self._init_backend()

        # Compute neighborhoods based on spatial_scale
        if self.spatial_scale == "searchlight":
            self.neighborhoods_ = compute_searchlight_neighborhoods(
                mask, radius=self.radius
            )
        elif self.spatial_scale == "roi":
            self.neighborhoods_ = _compute_roi_neighborhoods(self.roi_mask, mask)

        # Initialize storage for transforms and templates
        self.transforms_: dict[int, list[np.ndarray]] = {}
        self.template_: dict[int, np.ndarray] = {}

        # Fit local alignment for each neighborhood using batched processing
        if self.spatial_scale == "searchlight":
            n_regions = self.neighborhoods_.n_voxels
            region_type = "searchlight spheres"
        else:
            n_regions = self.neighborhoods_.n_parcels
            region_type = "parcels"
        logger.info(f"Fitting LocalAlignment with {n_regions} {region_type}")

        # Use batched iteration for memory efficiency
        batch_gen = self._batch_neighborhoods(
            self.neighborhoods_, n_subjects, n_samples
        )

        # Progress bar for total neighborhoods (opt-in via progress_bar)
        pbar = make_progress_bar(
            progress_bar=self.progress_bar,
            total=n_regions,
            desc=region_type.capitalize(),
            unit="regions",
        )

        # Determine parallelization strategy
        # CPU parallel: use joblib with numpy backend (each worker gets own numpy)
        # GPU: sequential processing with torch backend (GPU parallelizes internally)
        use_cpu_parallel = self.parallel == "cpu" and self.n_jobs != 1
        use_gpu = self.parallel == "gpu" and self.backend_.name.startswith("torch")

        for batch in batch_gen:
            if use_cpu_parallel and len(batch) > 1:
                # CPU parallel processing within batch (uses numpy, no backend passed)
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_one_neighborhood)(
                        region_id,
                        voxel_indices,
                        data,
                        self.method,
                        self.n_iter,
                        self.n_features,
                        None,  # Each worker uses numpy backend
                    )
                    for region_id, voxel_indices in batch
                )

                # Store results
                for region_id, transforms, template in results:
                    self.transforms_[region_id] = transforms
                    self.template_[region_id] = template

                pbar.update(len(batch))
            else:
                # Sequential processing (GPU mode or single-threaded)
                # Pass backend for GPU acceleration
                backend_to_use = self.backend_ if use_gpu else None
                for region_id, voxel_indices in batch:
                    _, transforms, template = _fit_one_neighborhood(
                        region_id,
                        voxel_indices,
                        data,
                        self.method,
                        self.n_iter,
                        self.n_features,
                        backend_to_use,
                    )
                    self.transforms_[region_id] = transforms
                    self.template_[region_id] = template
                    pbar.update(1)

            # Explicit cleanup after each batch for memory efficiency
            del batch

        pbar.close()
        logger.info("LocalAlignment fitting complete")
        return self

    def transform(self, data: list[np.ndarray]) -> list[np.ndarray]:
        """Apply local transforms to data.

        For the searchlight scale with center-only aggregation: each voxel uses
        the transform from the neighborhood where it was the center.

        For the roi scale: all voxels in each parcel use the same transform.

        Args:
            data (list[np.ndarray]): List of subject data arrays, each shape
                (n_voxels, n_samples).

        Returns:
            list[np.ndarray]: Aligned data for each subject, each shape
                (n_voxels, n_samples).
        """
        if self.transforms_ is None:
            raise ValueError("Model must be fit before transform")

        n_subjects = len(data)
        n_voxels, n_samples = data[0].shape

        if n_voxels != self.n_voxels_:
            raise ValueError(
                f"Data has {n_voxels} voxels but model was fit with {self.n_voxels_}"
            )

        # Determine parallelization strategy
        use_parallel = self.parallel == "cpu" and self.n_jobs != 1 and n_subjects > 1

        if use_parallel:
            # Parallel transform across subjects
            from joblib import Parallel, delayed

            def _transform_one_subject(subj_idx: int) -> np.ndarray:
                """Transform one subject's data."""
                subj_data = data[subj_idx]
                result = np.zeros((n_voxels, n_samples))

                for region_id, voxel_indices in self.neighborhoods_.iter_neighborhoods(
                    progress_bar=False
                ):
                    transforms = self.transforms_[region_id]
                    local_data = subj_data[voxel_indices, :]
                    transform = transforms[subj_idx]

                    if self.method == "procrustes":
                        aligned_local = transform @ local_data
                    else:
                        shared = transform.T @ local_data
                        aligned_local = transform @ shared

                    if self.aggregation == "center":
                        center_pos = np.where(voxel_indices == region_id)[0]
                        if len(center_pos) == 0:
                            center_pos = 0
                        else:
                            center_pos = center_pos[0]
                        result[region_id, :] = aligned_local[center_pos, :]
                    else:
                        result[voxel_indices, :] = aligned_local

                return result

            aligned = Parallel(n_jobs=self.n_jobs)(
                delayed(_transform_one_subject)(i) for i in range(n_subjects)
            )
        else:
            # Sequential transform
            aligned = [np.zeros((n_voxels, n_samples)) for _ in range(n_subjects)]

            for region_id, voxel_indices in self.neighborhoods_.iter_neighborhoods(
                progress_bar=self.progress_bar
            ):
                transforms = self.transforms_[region_id]

                for subj_idx, subj_data in enumerate(data):
                    local_data = subj_data[voxel_indices, :]
                    transform = transforms[subj_idx]

                    if self.method == "procrustes":
                        aligned_local = transform @ local_data
                    else:
                        shared = transform.T @ local_data
                        aligned_local = transform @ shared

                    if self.aggregation == "center":
                        center_pos = np.where(voxel_indices == region_id)[0]
                        if len(center_pos) == 0:
                            center_pos = 0
                        else:
                            center_pos = center_pos[0]
                        aligned[subj_idx][region_id, :] = aligned_local[center_pos, :]
                    else:
                        aligned[subj_idx][voxel_indices, :] = aligned_local

        return aligned

    def fit_transform(
        self, data: list[np.ndarray], mask: nib.Nifti1Image
    ) -> list[np.ndarray]:
        """Fit alignment and transform data in one step.

        Args:
            data (list[np.ndarray]): Subject data arrays, each of shape
                (n_voxels, n_samples).
            mask (nib.Nifti1Image): Brain mask defining the voxel space.

        Returns:
            list[np.ndarray]: Aligned data for each subject.
        """
        return self.fit(data, mask).transform(data)


__all__ = ["LocalAlignment", "RoiNeighborhoods"]
