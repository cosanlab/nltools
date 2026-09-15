"""A synthetic Haxby-like dataset, built the way `Simulator` builds data.

Despite living beside the fetchers in `nltools.datasets`, `load_haxby_example`
downloads no data: it injects condition-specific signal into a few spheres of
the package's 3 mm MNI brain mask (or, on request, into a tiny synthetic
volume). That makes it a simulator, so its implementation sits here.
"""

import numpy as np

from nltools.data.braindata import BrainData
from nltools.data.designmatrix import DesignMatrix
from nltools.io.events import events_to_dm


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

# Where each condition's response is injected, in MNI millimetres. The
# assignment follows the ventral-stream literature the Haxby experiment
# established: faces peak in the right fusiform face area, houses and other
# scene-like stimuli in the parahippocampal place area, animals just lateral
# and anterior to the face patch, and the man-made objects in the lateral
# occipital complex and posterior fusiform. Scrambled pictures carry no object
# information, so their response sits in early visual cortex instead.
_ROI_CENTERS_MNI = {
    "face": (40, -50, -20),  # right fusiform face area
    "house": (-26, -44, -10),  # left parahippocampal place area
    "cat": (-40, -50, -20),  # left fusiform, the animate counterpart of FFA
    "bottle": (46, -78, -6),  # right lateral occipital complex
    "scissors": (-46, -78, -6),  # left lateral occipital complex
    "shoe": (36, -66, -16),  # right posterior fusiform
    "chair": (-36, -66, -16),  # left posterior fusiform
    "scrambledpix": (0, -88, 2),  # early visual cortex
}

# No two centers are closer than 17 mm, so 8 mm spheres stay disjoint on the
# 3 mm grid and each covers 72-82 voxels.
_ROI_RADIUS_MM = 8.0

# Category membership, which is what gives the eight patterns a similarity
# structure worth looking at: a condition drives its own sphere fully and the
# spheres of its category-mates weakly, so animate conditions resemble each
# other, the four man-made objects resemble each other, and houses and
# scrambled pictures stand apart.
_CATEGORIES = {
    "face": "animate",
    "cat": "animate",
    "bottle": "object",
    "scissors": "object",
    "shoe": "object",
    "chair": "object",
    "house": "scene",
    "scrambledpix": "control",
}
_CATEGORY_CROSSTALK = 1.0 / 3.0

_TR_SECONDS = 2.5
_BLOCK_TRS = 6  # 15 s of stimulation, as in the real experiment's 24 s blocks
_REST_TRS = 3  # 7.5 s of rest, which keeps the design full rank
_N_TIMEPOINTS = (_BLOCK_TRS + _REST_TRS) * len(_HAXBY_CONDITIONS)  # 72 TRs
_REST_LABEL = "rest"

# A positive baseline keeps percent-signal-change scaling inside GLM fits
# (which divides by the voxel mean) numerically sane. The amplitude is a peak
# response of 3% against 1% white noise: high for real fMRI, but it is what
# lets a single 72-TR run put a contrast peak inside the intended sphere and
# decode the conditions from those spheres well above chance.
_BASELINE = 100.0
_NOISE_SD = 1.0
_SIGNAL_AMPLITUDE = 3.0

_GRID_SHAPE = (10, 10, 5)
_GRID_VOXEL_SIZE = 3.0


def _mask_image(space):
    """Return the mask image defining the voxel space of a simulated run."""
    import nibabel as nib

    if space == "mni":
        from nltools.templates import BrainSpaceConfig

        return nib.load(BrainSpaceConfig(template="default", resolution=3).mask)
    affine = np.diag([_GRID_VOXEL_SIZE] * 3 + [1.0]).astype(np.float32)
    return nib.Nifti1Image(np.ones(_GRID_SHAPE, dtype=np.float32), affine)


def _condition_rois(space, mask_img, rng):
    """Map each condition to a boolean mask over the in-mask voxel axis.

    On the MNI grid the regions are anatomical spheres from `create_sphere`; on
    the tiny synthetic grid they are disjoint random voxel clusters.
    """
    mask_bool = np.asarray(mask_img.dataobj) > 0
    n_voxels = int(mask_bool.sum())
    if space == "mni":
        from nltools.mask import create_sphere

        return {
            condition: np.asarray(
                create_sphere(
                    list(center), radius=_ROI_RADIUS_MM, mask=mask_img
                ).dataobj
            )[mask_bool]
            > 0
            for condition, center in _ROI_CENTERS_MNI.items()
        }
    voxels_per_condition = n_voxels // len(_HAXBY_CONDITIONS)
    voxel_order = rng.permutation(n_voxels)
    rois = {}
    for index, condition in enumerate(_HAXBY_CONDITIONS):
        cluster = voxel_order[
            index * voxels_per_condition : (index + 1) * voxels_per_condition
        ]
        roi = np.zeros(n_voxels, dtype=bool)
        roi[cluster] = True
        rois[condition] = roi
    return rois


def _response_weight(condition, roi_condition):
    """How strongly `condition` drives the sphere belonging to `roi_condition`."""
    if condition == roi_condition:
        return 1.0
    if _CATEGORIES[condition] == _CATEGORIES[roi_condition]:
        return _CATEGORY_CROSSTALK
    return 0.0


def _run_design(order):
    """Build one run's convolved `DesignMatrix` and its per-TR condition labels."""
    import pandas as pd

    trial_trs = _BLOCK_TRS + _REST_TRS
    events = pd.DataFrame(
        [
            {
                "onset": index * trial_trs * _TR_SECONDS,
                "duration": _BLOCK_TRS * _TR_SECONDS,
                "trial_type": condition,
            }
            for index, condition in enumerate(order)
        ]
    )
    dm = (
        DesignMatrix(
            events_to_dm(
                events,
                run_length=_N_TIMEPOINTS,
                sampling_freq=1.0 / _TR_SECONDS,
            ),
            sampling_freq=1.0 / _TR_SECONDS,
        )
        .convolve()
        .add_poly(0)
    )

    labels = np.full(_N_TIMEPOINTS, _REST_LABEL, dtype=object)
    for index, condition in enumerate(order):
        start = index * trial_trs
        labels[start : start + _BLOCK_TRS] = condition
    return dm, labels


def _simulate_run(dm, rois, n_voxels, rng):
    """Return one run's (n_timepoints, n_voxels) array of noise plus ROI signal."""
    data = _BASELINE + _NOISE_SD * rng.standard_normal(
        (_N_TIMEPOINTS, n_voxels)
    ).astype(np.float32)
    for condition in rois:
        regressor = np.asarray(dm[f"{condition}_c0"], dtype=np.float32)
        regressor = (regressor / regressor.max())[:, None]
        for roi_condition, roi in rois.items():
            weight = _response_weight(condition, roi_condition)
            if weight:
                data[:, roi] += _SIGNAL_AMPLITUDE * weight * regressor
    return data


def load_haxby_example(
    n_runs=1, *, space="mni", block_order="independent", random_state=42
):
    """Load a synthetic Haxby-like dataset on the MNI grid, entirely in-memory.

    The quickest way to try nltools: nothing is downloaded beyond the MNI
    template `fetch_resource` already caches, and a run takes a second or two
    to build. Returns paired lists of `BrainData` and `DesignMatrix`, one entry
    per run.

    Each run is a randomized block design of the eight conditions from the real
    Haxby 2001 object-recognition experiment (face, house, cat, bottle,
    scissors, shoe, chair, scrambledpix): 15 s of stimulation followed by 7.5 s
    of rest, eight times, at TR = 2.5 s for 72 TRs. Every condition drives an
    8 mm sphere at an anatomically plausible location — faces in the right
    fusiform face area, houses in the left parahippocampal place area, animals
    in the left fusiform, man-made objects in lateral occipital and posterior
    fusiform cortex, and scrambled pictures in early visual cortex — with a 3%
    peak response against 1% white noise. A condition also drives its
    category-mates' spheres at a third of that, so the eight response patterns
    carry an animate / man-made / scene / control similarity structure. Each run
    draws its own block order by default; pass `block_order='shared'` when the
    runs have to line up TR by TR, as they do for intersubject correlation,
    alignment and anything else that compares runs timepoint against timepoint.

    Args:
        n_runs (int): Number of runs to generate. Default 1.
        space (str): Voxel space of the data. `'mni'` (default) is the package's
            3 mm MNI152 mask, so every plotting method, `apply_mask` and
            `extract_roi` work on the result. `'grid'` is a 10 x 10 x 5
            synthetic volume with random condition clusters, for tests that need
            construction in milliseconds and never plot.
        block_order (str): How the condition order varies across runs.
            `'independent'` (default) draws a fresh order per run; `'shared'`
            draws one order and gives it to every run, so the runs are
            comparable TR by TR. Noise differs per run either way.
        random_state (int | None): Seed for reproducible output. Default 42.

    Returns:
        tuple: `(list[BrainData], list[DesignMatrix])`, each of length `n_runs`.
            Each `BrainData` carries a `.Y` with a `condition` column (the eight
            condition names plus `'rest'`) and a `run` column. The
            `DesignMatrix` columns are the eight condition names suffixed with
            `_c0` (HRF-convolved boxcars).

    Raises:
        ValueError: If `space` is not `'mni'` or `'grid'`, or `block_order` is not
            `'independent'` or `'shared'`.

    Examples:
        ```python
        from nltools.datasets import load_haxby_example
        from nltools.mask import create_sphere

        brain_data, design_matrices = load_haxby_example()
        data, dm = brain_data[0], design_matrices[0]

        # A first-level contrast; the face patch is visible without a threshold
        data.fit("glm", X=dm)
        data.compute_contrasts("face_c0 - house_c0").plot(method="glass")

        # Decode faces from houses inside the regions that carry the signal
        ventral_temporal = create_sphere(
            [[40, -50, -20], [-26, -44, -10]], radius=8, mask=data.mask
        )
        trs = data[data.Y["condition"].is_in(["face", "house"])]
        trs.apply_mask(ventral_temporal).predict(y="condition", cv=3).mean_score
        # → 0.83
        ```

    Note:
        Decoding and representational similarity need a region, not the whole
        brain: one run labels only twelve TRs per pair of conditions, and
        71,020 voxels of noise swamp the ~150 that respond. Mask down to the
        response spheres — `create_sphere` with the centers above — before
        `predict` or `distance`, the way ventral-temporal MVPA is done on the
        real dataset.
    """
    import polars as pl

    if space not in ("mni", "grid"):
        raise ValueError(f"space must be 'mni' or 'grid', got {space!r}")
    if block_order not in ("independent", "shared"):
        raise ValueError(
            f"block_order must be 'independent' or 'shared', got {block_order!r}"
        )

    rng = np.random.default_rng(random_state)
    mask_img = _mask_image(space)
    n_voxels = int((np.asarray(mask_img.dataobj) > 0).sum())
    rois = _condition_rois(space, mask_img, rng)

    # Drawn once so every run gets the same sequence; 'independent' draws inside
    # the loop instead, leaving the default run's RNG stream untouched.
    shared_order = (
        list(rng.permutation(_HAXBY_CONDITIONS)) if block_order == "shared" else None
    )

    brain_data_list = []
    design_matrix_list = []
    for run in range(n_runs):
        order = (
            shared_order
            if shared_order is not None
            else list(rng.permutation(_HAXBY_CONDITIONS))
        )
        dm, labels = _run_design(order)
        data = _simulate_run(dm, rois, n_voxels, rng)
        labels_and_run = pl.DataFrame(
            {
                "condition": labels.astype(str),
                "run": np.full(_N_TIMEPOINTS, run, dtype=int),
            }
        )
        brain_data_list.append(
            BrainData(data, mask=mask_img, Y=labels_and_run, verbose=0)
        )
        design_matrix_list.append(dm)

    return brain_data_list, design_matrix_list
