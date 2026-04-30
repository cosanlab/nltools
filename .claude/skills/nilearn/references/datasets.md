# nilearn.datasets — Automatic Dataset Fetching

Helper functions to download NeuroImaging datasets (atlases, brain templates, example fMRI/structural datasets, statistical maps, and coordinate atlases). Functions prefixed `fetch_*` download to a local cache on first call; `load_*` helpers return data bundled with nilearn (no network).

**Source:** https://nilearn.github.io/dev/modules/datasets.html

## When to use
- You need a brain atlas (parcellation or coordinate ROIs) for masking, connectivity, or labeling.
- You need an MNI152 or fsaverage template / brain mask as a reference image.
- You want a small example fMRI dataset (Haxby, ADHD, SPM auditory) to prototype an analysis.
- You want statistical maps from Neurovault / OpenNeuro / Brainomics for replication or meta-analysis.
- You need to inspect or override where nilearn caches downloaded data.

## Quick reference

### Templates & masks (bundled, no download)
| Function | Returns |
|---|---|
| `load_mni152_template(resolution=2)` | MNI152 skullstripped T1 template |
| `load_mni152_gm_template(resolution)` | MNI152 grey-matter template |
| `load_mni152_wm_template(resolution)` | MNI152 white-matter template |
| `load_mni152_brain_mask(resolution, threshold)` | MNI152 whole-brain mask |
| `load_mni152_gm_mask(resolution, threshold, ...)` | MNI152 grey-matter mask |
| `load_mni152_wm_mask(resolution, threshold, ...)` | MNI152 white-matter mask |
| `load_fsaverage(mesh, data_dir)` | fsaverage mesh (both hemispheres) as `PolyMesh` |
| `load_fsaverage_data(mesh, mesh_type, ...)` | fsaverage data on a mesh as a `SurfaceImage` |
| `load_sample_motor_activation_image()` | Single functional image showing motor activations |

### Templates (downloaded on first call)
| Function | Description |
|---|---|
| `fetch_icbm152_2009(data_dir, url, resume, ...)` | Download and load the ICBM152 template (dated 2009) |
| `fetch_icbm152_brain_gm_mask(data_dir, ...)` | Download ICBM152 template, then load the GM mask |
| `fetch_surf_fsaverage(mesh, data_dir)` | Download a Freesurfer fsaverage surface (legacy dict API) |

### Deterministic atlases (downloaded on first call)
| Function | Description | Key params |
|---|---|---|
| `fetch_atlas_aal(version, data_dir, url, ...)` | Download and return the AAL template for SPM 12 | `version` (`'SPM12'`, `'3v2'`); 117 or 167 regions |
| `fetch_atlas_basc_multiscale_2015(data_dir, ...)` | Download multiscale functional brain parcellations | resolution in `[7, 12, 20, 36, 64, 122, 197, 325, 444]` |
| `fetch_atlas_destrieux_2009(lateralized, ...)` | Download Destrieux cortical deterministic atlas (2009) | `lateralized` (bool); 148 regions |
| `fetch_atlas_harvard_oxford(atlas_name, ...)` | Load Harvard-Oxford parcellations from FSL | `atlas_name` (e.g. `'cort-maxprob-thr25-2mm'`) |
| `fetch_atlas_juelich(atlas_name, data_dir, ...)` | Load Juelich parcellations from FSL | `atlas_name`; ~62 regions |
| `fetch_atlas_pauli_2017(atlas_type, ...)` | Download the Pauli et al. (2017) subcortical atlas | `atlas_type` (`'deterministic'`/`'probabilistic'`); 16 regions |
| `fetch_atlas_schaefer_2018(n_rois, ...)` | Download Schaefer 2018 cortical parcellation | `n_rois` (100–1000), `yeo_networks` (7 or 17), `resolution_mm` (1 or 2) |
| `fetch_atlas_surf_destrieux(data_dir, url, ...)` | Download Destrieux et al. 2010 cortical surface atlas | 75 regions per hemisphere |
| `fetch_atlas_talairach(level_name, ...)` | Download the Talairach deterministic atlas | `level_name` (`'hemisphere'`, `'lobe'`, `'gyrus'`, `'tissue'`, `'ba'`) |
| `fetch_atlas_yeo_2011(data_dir, url, ...)` | Download Yeo 2011 parcellation | 7 or 17 networks |

### Probabilistic atlases (downloaded on first call)
| Function | Description | Key params |
|---|---|---|
| `fetch_atlas_allen_2011(data_dir, url, ...)` | Allen and MIALAB ICA Probabilistic atlas (2011) | 75 regions |
| `fetch_atlas_craddock_2012(data_dir, url, ...)` | Craddock 2012 parcellation | spatially-constrained spectral clustering |
| `fetch_atlas_difumo(dimension, ...)` | Fetch DiFuMo brain atlas | `dimension` in `[64, 128, 256, 512, 1024]`, `resolution_mm` (2 or 3) |
| `fetch_atlas_harvard_oxford(atlas_name, ...)` | Harvard-Oxford parcellations from FSL (also probabilistic variants) | `atlas_name` (e.g. `'cort-prob-2mm'`) |
| `fetch_atlas_juelich(atlas_name, data_dir, ...)` | Juelich parcellations from FSL (also probabilistic variants) | `atlas_name` |
| `fetch_atlas_msdl(data_dir, url, resume, ...)` | MSDL brain probabilistic atlas | 39 regions |
| `fetch_atlas_pauli_2017(atlas_type, ...)` | Pauli et al. (2017) atlas (probabilistic variant) | `atlas_type='probabilistic'` |
| `fetch_atlas_smith_2009(data_dir, url, ...)` | Smith ICA and BrainMap probabilistic atlas (2009) | `dimension` in `[10, 20, 70]` |

### Coordinate atlases
| Function | Description |
|---|---|
| `fetch_coords_dosenbach_2010(ordered_regions)` | Load the Dosenbach et al. 160 ROIs |
| `fetch_coords_power_2011()` | Download and load the Power et al. 264-ROI atlas |
| `fetch_coords_seitzman_2018(ordered_regions)` | Load the Seitzman et al. 300 ROIs |

### Surface atlases
| Function | Description |
|---|---|
| `fetch_atlas_surf_destrieux(data_dir, url, ...)` | Destrieux 2010 cortical surface atlas |
| `fetch_surf_fsaverage(mesh, data_dir)` | fsaverage Freesurfer surface (legacy dict format) |
| `load_fsaverage(mesh, data_dir)` | fsaverage as a `PolyMesh` (modern API) |
| `load_fsaverage_data(mesh, mesh_type, ...)` | fsaverage data as a `SurfaceImage` |

### Example datasets (fMRI / preprocessed)
| Function | Description |
|---|---|
| `fetch_abide_pcp(data_dir, n_subjects, ...)` | Fetch ABIDE dataset |
| `fetch_adhd(n_subjects, data_dir, url, ...)` | Download ADHD resting-state dataset |
| `fetch_development_fmri(n_subjects, ...)` | Movie-watching brain development dataset (fMRI) |
| `fetch_ds000030_urls(data_dir, verbose)` | URLs for files in the ds000030 BIDS dataset |
| `fetch_fiac_first_level(data_dir, verbose)` | First-level FIAC fMRI dataset (2 runs) |
| `fetch_haxby(data_dir, subjects, ...)` | Complete Haxby dataset |
| `fetch_language_localizer_demo_dataset(...)` | Language localizer demo dataset (BIDS) |
| `fetch_localizer_first_level(data_dir, verbose)` | First-level localizer fMRI dataset |
| `fetch_miyawaki2008(data_dir, url, resume, ...)` | Miyawaki et al. 2008 dataset (153MB) |
| `fetch_spm_auditory(data_dir, data_name, ...)` | SPM auditory single-subject data |
| `fetch_spm_multimodal_fmri(data_dir, ...)` | Multi-modal Face Dataset |
| `fetch_surf_nki_enhanced(n_subjects, ...)` | NKI enhanced resting-state surface dataset (fsaverage5) |
| `load_nki(mesh, mesh_type, n_subjects, ...)` | Load NKI enhanced surface data into a surface object |

### Statistical maps & derivatives
| Function | Description |
|---|---|
| `fetch_localizer_button_task(data_dir, verbose)` | Left vs right button-press contrast maps from the localizer |
| `fetch_localizer_calculation_task(...)` | Calculation task contrast maps from the localizer |
| `fetch_localizer_contrasts(contrasts, ...)` | Brainomics/Localizer dataset (94 subjects) |
| `fetch_megatrawls_netmats(dimensionality, ...)` | Network matrices from the MegaTrawls release in HCP |
| `fetch_mixed_gambles(n_subjects, data_dir, ...)` | Jimura "mixed gambles" dataset |
| `fetch_oasis_vbm(n_subjects, ...)` | OASIS "cross-sectional MRI" dataset (416 subjects) |
| `fetch_neurovault_auditory_computation_task(...)` | Neurovault contrast: mental subtraction with auditory instructions |

### General fetchers & cache utilities
| Function | Description |
|---|---|
| `fetch_neurovault(max_images, ...)` | Download data from neurovault.org matching given criteria |
| `fetch_neurovault_ids(collection_ids, ...)` | Download specific images/collections from neurovault.org |
| `fetch_openneuro_dataset(urls, data_dir, ...)` | Download an OpenNeuro BIDS dataset |
| `patch_openneuro_dataset(file_list)` | Add symlinks for files not following BIDS conventions |
| `select_from_index(urls, inclusion_filters, ...)` | Subset Neurovault/OpenNeuro URLs with filters |
| `get_data_dirs(data_dir)` | Return the directories nilearn searches for cached data |

## Common usage patterns

```python
# Load a parcellation atlas with labels
from nilearn.datasets import fetch_atlas_schaefer_2018
atlas = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
atlas.maps     # path to / Niimg of the parcellation
atlas.labels   # array of region names
```

```python
# Bundled MNI152 templates and masks (no network)
from nilearn.datasets import (
    load_mni152_template, load_mni152_brain_mask,
    load_mni152_gm_mask, load_mni152_wm_mask,
)
template = load_mni152_template(resolution=2)
brain    = load_mni152_brain_mask(resolution=2)
gm       = load_mni152_gm_mask(resolution=2)
wm       = load_mni152_wm_mask(resolution=2)
```

```python
# Coordinate atlases for sphere-based connectivity
from nilearn.datasets import fetch_coords_power_2011, fetch_coords_dosenbach_2010
power = fetch_coords_power_2011()        # 264 ROIs
dose  = fetch_coords_dosenbach_2010()    # 160 ROIs
coords = list(zip(power.rois['x'], power.rois['y'], power.rois['z']))
```

```python
# Modern surface API (fsaverage)
from nilearn.datasets import load_fsaverage, load_fsaverage_data
fsavg5 = load_fsaverage('fsaverage5')         # PolyMesh
sulc   = load_fsaverage_data('fsaverage5', data_type='sulcal')
```

```python
# Example fMRI dataset for prototyping
from nilearn.datasets import fetch_haxby, fetch_development_fmri
haxby = fetch_haxby(subjects=[1])
dev   = fetch_development_fmri(n_subjects=5)   # small movie-watching subset
```

```python
# Probabilistic atlas (DiFuMo) plus Harvard-Oxford for masking
from nilearn.datasets import fetch_atlas_difumo, fetch_atlas_harvard_oxford
difumo = fetch_atlas_difumo(dimension=64, resolution_mm=2)
ho     = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
```

```python
# Inspect or override the cache directory
from nilearn.datasets import get_data_dirs
print(get_data_dirs())                              # default search path
atlas = fetch_atlas_schaefer_2018(data_dir='/data/nilearn_cache')
```

## Gotchas
1. `fetch_*` downloads from the internet on first call; subsequent calls hit a cache (default `~/nilearn_data`, override with `data_dir=` or env var `NILEARN_SHARED_DATA` / `NILEARN_DATA`). `get_data_dirs()` shows the search path.
2. `load_*` loads bundled data instantly — no network. Prefer `load_mni152_template` over downloading a fresh copy.
3. `fetch_atlas_*` functions return a `Bunch` (dict-like). The image is usually `.maps` and labels `.labels`, but attribute names vary by atlas (e.g. coordinate atlases use `.rois`). Inspect the returned object or check the docstring.
4. Some example datasets are large multi-GB downloads (Haxby ~300MB+, ABIDE PCP, OASIS, OpenNeuro full datasets). Pass `n_subjects=` where supported to limit, and warn the user before fetching.
5. `fetch_atlas_harvard_oxford` and `fetch_atlas_juelich` exist in both deterministic and probabilistic flavours — pick by `atlas_name` (e.g. `'cort-maxprob-thr25-2mm'` vs `'cort-prob-2mm'`).
6. Atlas templates do not always match nilearn's default plotting MNI template (see warnings on the deterministic/probabilistic atlas pages). Misalignment can produce a parcellation that visually appears smaller/larger than the underlay; do not use a misregistered atlas with maskers.
7. Surface API: `fetch_surf_fsaverage()` returns the legacy dict format; `load_fsaverage()` returns the modern `PolyMesh`. Prefer the latter for new code.
8. `fetch_neurovault` / `fetch_openneuro_dataset` can pull large numbers of files — always combine with `select_from_index` (or `inclusion_filters` / `exclusion_filters`) and a `max_images=` cap.
9. `patch_openneuro_dataset` is sometimes required after `fetch_openneuro_dataset` to make filenames BIDS-compliant before downstream tools accept them.
10. `fetch_atlas_schaefer_2018` accepts `n_rois` only from `{100, 200, ..., 1000}` and `yeo_networks` only `7` or `17` — other values raise.

## See also
- `references/maskers.md` — `NiftiLabelsMasker` / `NiftiMapsMasker` consume atlas `.maps` images directly
- `references/surface.md` — surface objects returned by `load_fsaverage*` and `load_nki`
- `references/interfaces.md` — `nilearn.interfaces.fmriprep.load_confounds` for fMRIPrep-derived data
- https://nilearn.github.io/dev/modules/datasets.html
