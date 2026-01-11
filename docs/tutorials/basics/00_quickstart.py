# %% [markdown]
# # Quickstart
#
# Get started with nltools in under a minute.

# %%
# Load example fMRI data
from nltools.datasets import fetch_haxby

data, design = fetch_haxby(n_subjects=1)
brain = data[0]  # First run

print(f"Shape: {brain.shape} (timepoints × voxels)")

# %%
# Plot the mean brain
brain.mean().plot()

# %%
# Basic operations work like numpy/pandas
first_10 = brain[:10]  # Slice timepoints
scaled = brain * 2 + 100  # Arithmetic
mean_ts = brain.mean(axis=1)  # Mean per timepoint

# %%
# Convert to NIfTI for use with other tools
nifti = brain.to_nifti()
print(f"NIfTI shape: {nifti.shape}")

# %% [markdown]
# ## Next Steps
#
# - **[BrainData Basics](01_brain_data)**: Complete guide to the BrainData class
# - **[DesignMatrix Basics](02_design_matrix)**: Building experimental designs
# - **[Adjacency Basics](03_adjacency)**: Connectivity and similarity matrices
# - **[GLM Workflow](../workflows/01_glm)**: First-level fMRI analysis
