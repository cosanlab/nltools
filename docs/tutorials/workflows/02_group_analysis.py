# %% [markdown]
# # Group Analysis
#
# This tutorial covers the standard fMRI group analysis workflow:
# first-level GLM for each subject, then second-level statistics across subjects.

# %%
import matplotlib

matplotlib.use("Agg")

from scipy.stats import ttest_1samp

from nltools.datasets import fetch_haxby
from nltools.stats import fdr
from nltools.utils import concatenate

# %% [markdown]
# ## Load Data

# %%
# Load Haxby dataset (use first 3 subjects for speed)
print("Loading Haxby dataset...")
all_data, all_dm = fetch_haxby(n_subjects="all", verbose=1)

n_subjects = 3
all_subjects = all_data[:n_subjects]
all_subjects_dm = all_dm[:n_subjects]

print(f"\nUsing {len(all_subjects)} subjects")

# %% [markdown]
# ## First-Level: Fit GLM for Each Subject

# %%
# Fit GLM for each subject and compute contrasts
contrasts_face_vs_house = []

for i, (subj_runs, subj_dm) in enumerate(zip(all_subjects, all_subjects_dm)):
    print(f"Processing subject {i + 1}...")

    data = subj_runs[0].copy()
    dm = subj_dm[0]

    # Add nuisance regressors
    dm_filt = dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)

    # Fit GLM
    data.fit(model="glm", X=dm_filt)

    # Compute contrast
    contrasts_face_vs_house.append(data.compute_contrasts("face - house"))

print(f"\nExtracted contrasts for {len(contrasts_face_vs_house)} subjects")

# %% [markdown]
# ## Second-Level: Group T-Test

# %%
# Stack contrasts and run voxel-wise t-test
contrast_group = concatenate(contrasts_face_vs_house)
print(f"Group data shape: {contrast_group.shape}")

t_values, p_values = ttest_1samp(contrast_group.data, 0, axis=0)
print(f"T-test computed for {len(t_values)} voxels")

# %% [markdown]
# ## Visualize Results

# %%
# Create BrainData for visualization
t_map = contrast_group[0].copy()
t_map.data = t_values.reshape(1, -1)

print("T-statistic: Face > House")
t_map.plot(title="Face > House (t-statistic)")

# %%
# Mean contrast across subjects
print("\nMean contrast (Face - House):")
contrast_group.mean().plot(title="Mean (Face - House)")

# %% [markdown]
# ## Thresholding for Multiple Comparisons
#
# When testing thousands of voxels, we need to correct for multiple comparisons.
# Common approaches:
# - **FDR**: Controls proportion of false positives among significant results
# - **Bonferroni**: Controls any false positive (very conservative)

# %%
# FDR correction (q < 0.05)
fdr_threshold = fdr(p_values, q=0.05)
print(f"FDR threshold (q < 0.05): p < {fdr_threshold:.6f}")

if fdr_threshold > 0:
    sig_mask_fdr = p_values < fdr_threshold
    n_sig_fdr = sig_mask_fdr.sum()
    print(f"FDR significant voxels: {n_sig_fdr}")

    # Create thresholded map
    t_map_fdr = t_map.copy()
    t_map_fdr.data[0, ~sig_mask_fdr] = 0
    t_map_fdr.plot(title="Face > House (FDR q < 0.05)")
else:
    print("No voxels survive FDR correction")

# %%
# Bonferroni correction (conservative)
n_voxels = len(p_values)
bonf_threshold = 0.05 / n_voxels
sig_mask_bonf = p_values < bonf_threshold
print(f"\nBonferroni threshold: p < {bonf_threshold:.2e}")
print(f"Bonferroni significant voxels: {sig_mask_bonf.sum()}")

# %% [markdown]
# ## Summary
#
# | Step | Method |
# |------|--------|
# | First-level | `data.fit(model='glm', X=dm)` |
# | Compute contrast | `data.compute_contrasts("face - house")` |
# | Stack subjects | `concatenate(contrasts)` |
# | Group t-test | `scipy.stats.ttest_1samp()` |
# | FDR correction | `nltools.stats.fdr(p_values, q=0.05)` |
