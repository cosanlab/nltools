# nilearn.interfaces — External Tool Integration

Adapters for loading components from other neuroimaging tools. Three submodules: `nilearn.interfaces.bids` (BIDS dataset traversal), `nilearn.interfaces.fmriprep` (load fMRIPrep confounds with denoising strategies), and `nilearn.interfaces.fsl` (parse FSL design matrices).

**Source:** https://nilearn.github.io/dev/modules/interfaces.html

## Inventory

### nilearn.interfaces.bids
| Function | Purpose |
|---|---|
| `get_bids_files(main_path, ...)` | Search for files in a BIDS dataset following given constraints. |
| `parse_bids_filename(img_path)` | Return dictionary of parsed BIDS entities from a file path. |

### nilearn.interfaces.fmriprep
| Function | Purpose |
|---|---|
| `load_confounds(img_files, strategy=..., ...)` | Load fMRIPrep confounds with granular control. |
| `load_confounds_strategy(img_files, denoise_strategy=..., ...)` | Use a preset denoising strategy. |

### nilearn.interfaces.fsl
| Function | Purpose |
|---|---|
| `get_design_from_fslmat(fsl_design_matrix_path)` | Extract design matrix DataFrame from an FSL `.mat` file. |

## load_confounds

```python
from nilearn.interfaces.fmriprep import load_confounds

confounds, sample_mask = load_confounds(
    img_files,                                # path(s) to *_desc-preproc_bold.nii.gz
    strategy=('motion', 'high_pass', 'wm_csf'),
    motion='full',                            # 'basic'|'power2'|'derivatives'|'full'
    scrub=5,                                  # min N continuous volumes between censored
    fd_threshold=0.5,                         # framewise displacement (mm)
    std_dvars_threshold=1.5,
    wm_csf='basic',                           # 'basic'|'power2'|'derivatives'|'full'
    global_signal='basic',
    compcor='anat_combined',                  # 'anat_combined'|'anat_separated'|'temporal'|'temporal_anat_combined'|'temporal_anat_separated'
    n_compcor='all',
    ica_aroma='full',                         # 'full'|'basic'
    demean=True,
)
```

Strategy components (order in tuple does not matter):
- `'motion'` — translation/rotation parameters.
- `'high_pass'` — fMRIPrep cosine basis high-pass regressors.
- `'wm_csf'` — anatomical compartment signals.
- `'global_signal'` — whole-brain mean signal.
- `'compcor'` — anatomical or temporal CompCor components.
- `'ica_aroma'` — AROMA noise components (requires the `desc-aromaNonAggr` derivative).
- `'scrub'` — flags volumes via FD/DVARS for censoring.
- `'tedana'` — multi-echo denoising regressors.
- `'non_steady_state'` — non-steady-state outlier volumes.

Returns `(confounds_df, sample_mask)` per input file (a list when multiple files are passed). `sample_mask` is the indices of volumes to **keep** — pass it to `FirstLevelModel.fit(sample_masks=...)` or to `signal.clean(sample_mask=...)`.

## load_confounds_strategy

```python
from nilearn.interfaces.fmriprep import load_confounds_strategy

confounds, sample_mask = load_confounds_strategy(
    img_files,
    denoise_strategy='simple',  # 'simple'|'scrubbing'|'compcor'|'ica_aroma'|'tedana'
    motion='full',              # most strategies forward this through
    wm_csf='basic',             # for 'simple', 'scrubbing'
    global_signal=None,         # add 'basic' / 'derivatives' / 'full' to include GS
    scrub=5, fd_threshold=0.5,  # for 'scrubbing'
    n_compcor='all',            # for 'compcor'
)
```

Preset strategies:
- `'simple'` — motion + WM/CSF + high_pass + non_steady_state (Ciric Set 2 baseline).
- `'scrubbing'` — `simple` + scrub.
- `'compcor'` — motion + CompCor + high_pass + non_steady_state.
- `'ica_aroma'` — motion + AROMA non-aggressive + WM/CSF + high_pass.
- `'tedana'` — multi-echo tedana denoising regressors.

## BIDS

```python
from nilearn.interfaces.bids import get_bids_files, parse_bids_filename

# Find all BOLD files for sub-01, task-rest
files = get_bids_files(
    main_path='/data/bids',
    file_tag='bold',
    file_type='nii.gz',
    sub_label='01',
    filters=[('task', 'rest'), ('space', 'MNI152NLin2009cAsym')],
    sub_folder=True,
)

# Parse a single filename
entities = parse_bids_filename(
    'sub-01_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
)
# {'sub': '01', 'task': 'rest', 'run': '1', 'space': 'MNI152NLin2009cAsym', ...}
```

## FSL

```python
from nilearn.interfaces.fsl import get_design_from_fslmat
dm = get_design_from_fslmat('/path/to/design.mat')   # pandas DataFrame
```

## Common patterns

End-to-end fMRIPrep -> first-level GLM:

```python
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import FirstLevelModel

confounds, sample_mask = load_confounds(
    fmri_img,
    strategy=('motion', 'high_pass', 'wm_csf', 'scrub'),
    motion='full', scrub=5, fd_threshold=0.5,
)

flm = FirstLevelModel(t_r=2.0, hrf_model='glover', smoothing_fwhm=6)
flm.fit(fmri_img, events=events_df,
        confounds=confounds, sample_masks=sample_mask)
```

Preset denoising for connectivity:

```python
from nilearn.interfaces.fmriprep import load_confounds_strategy

confounds, sample_mask = load_confounds_strategy(
    fmri_img, denoise_strategy='scrubbing', global_signal='basic',
)

ts = masker.fit_transform(fmri_img, confounds=confounds, sample_mask=sample_mask)
```

Discover BIDS files for a task:

```python
from nilearn.interfaces.bids import get_bids_files

bolds = get_bids_files('/data/bids', file_tag='bold', file_type='nii.gz',
                        filters=[('task', 'nback'), ('desc', 'preproc')])
```

## Gotchas

- `load_confounds` returns `(confounds_df, sample_mask)` — always unpack. Forgetting `sample_mask` silently disables scrubbing.
- Pass `sample_mask` to `FirstLevelModel.fit(sample_masks=...)` (note the plural for first-level).
- `'ica_aroma'` requires the AROMA-cleaned BOLD file (`desc-smoothAROMAnonaggr_bold.nii.gz`); the regular `desc-preproc_bold.nii.gz` will not work.
- `compcor` defaults pull anatomical CompCor components — for temporal CompCor pass `compcor='temporal'`.
- When using `load_confounds_strategy`, do **not** also include `'high_pass'` in your GLM (it is already in the confounds); set `FirstLevelModel(high_pass=None)`.
- `get_bids_files` does not validate the dataset — it just globs filenames matching entities. Use `pybids` for validation.

## See also

- `nilearn.glm.first_level.first_level_from_bids` — wraps BIDS discovery + GLM setup.
- `nilearn.signal.clean` — accepts the same `sample_mask` for time-series cleaning.
- `nilearn.image.clean_img` — 4D-image equivalent.
- https://nilearn.github.io/dev/modules/interfaces.html
