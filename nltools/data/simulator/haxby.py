"""A synthetic Haxby-like dataset, built the way `Simulator` builds data.

Despite living beside the fetchers in `nltools.datasets`, `load_haxby_example`
downloads nothing: it injects condition-specific signal into a tiny seeded
volume. That makes it a simulator, so its implementation sits here.
"""

from nltools.data.braindata import BrainData
from nltools.data.designmatrix import DesignMatrix
from nltools.data.designmatrix.io import events_to_dm


_HAXBY_CONDITIONS = (
    "face",
    "house",
    "cat",
    "bottle",
    "scissors",
    "shoe",
    "chair",
    "scrambledpix",
)


def load_haxby_example(n_runs=1, random_state=42):
    """Load a small synthetic Haxby-like dataset, entirely in-memory.

    Returns paired lists of `BrainData` and `DesignMatrix`, one entry per
    run, generated from a tiny synthetic volume (10 x 10 x 5 = 500 voxels)
    with condition-specific signal injected into disjoint voxel clusters.
    No network I/O, no disk I/O, no nilearn fetcher dependency. Runs in
    well under a second.

    Intended for tutorials, documentation examples, and tests where
    downloading a real fMRI dataset is impractical. The
    eight conditions match the real Haxby 2001 object-recognition experiment
    (face, house, cat, bottle, scissors, shoe, chair, scrambledpix), arranged
    in a randomized 9-TR block design with TR=2.5s.

    Args:
        n_runs (int): Number of runs to generate. Default 1.
        random_state (int | None): Seed for reproducible output. Default 42.

    Returns:
        tuple: `(list[BrainData], list[DesignMatrix])`, each of length `n_runs`.
            The DesignMatrix columns are the eight condition names suffixed
            with `_c0` (HRF-convolved boxcars).

    Examples:
        ```python
        from nltools.datasets import load_haxby_example

        brain_data, design_matrices = load_haxby_example()
        data, dm = brain_data[0], design_matrices[0]
        data.shape  # → (72, 500)
        "face_c0" in dm.columns  # → True
        ```
    """
    import numpy as np
    import pandas as pd
    import nibabel as nib

    rng = np.random.default_rng(random_state)

    TR = 2.5
    block_tr = 9  # 22.5s blocks, same order of magnitude as real Haxby
    n_conditions = len(_HAXBY_CONDITIONS)
    n_timepoints = block_tr * n_conditions  # 72 TRs per run
    spatial_shape = (10, 10, 5)
    n_voxels = int(np.prod(spatial_shape))
    affine = np.diag([3.0, 3.0, 3.0, 1.0]).astype(np.float32)
    mask_img = nib.Nifti1Image(np.ones(spatial_shape, dtype=np.float32), affine)

    brain_data_list = []
    design_matrix_list = []

    for _ in range(n_runs):
        order = list(rng.permutation(_HAXBY_CONDITIONS))
        events_df = pd.DataFrame(
            [
                {
                    "onset": i * block_tr * TR,
                    "duration": block_tr * TR,
                    "trial_type": cond,
                }
                for i, cond in enumerate(order)
            ]
        )
        dm_data = events_to_dm(
            events_df,
            run_length=n_timepoints,
            sampling_freq=1.0 / TR,
        )
        dm = DesignMatrix(dm_data, sampling_freq=1.0 / TR).convolve()

        # Positive BOLD-like baseline intensity + voxelwise Gaussian noise,
        # plus signal injected into a disjoint voxel cluster per condition
        # so contrasts produce real spatial patterns. The positive baseline
        # is required so percent-signal-change scaling inside GLM fits
        # (which divides by the voxel mean) stays numerically sane.
        baseline = 100.0
        noise_sd = 1.0
        signal_strength = 5.0  # BOLD units → clearly visible pattern in plots
        data_flat = (
            baseline + noise_sd * rng.standard_normal((n_voxels, n_timepoints))
        ).astype(np.float32)
        voxels_per_cond = n_voxels // n_conditions
        voxel_order = rng.permutation(n_voxels)
        for c_idx, cond in enumerate(_HAXBY_CONDITIONS):
            cond_col = f"{cond}_c0"
            if cond_col not in dm.columns:
                continue
            regressor = np.asarray(dm[cond_col], dtype=np.float32)
            cluster = voxel_order[
                c_idx * voxels_per_cond : (c_idx + 1) * voxels_per_cond
            ]
            data_flat[cluster, :] += signal_strength * regressor

        data_4d = data_flat.reshape((*spatial_shape, n_timepoints))
        img = nib.Nifti1Image(data_4d, affine)
        brain_data_list.append(BrainData(img, mask=mask_img, verbose=0))
        design_matrix_list.append(dm)

    return brain_data_list, design_matrix_list
