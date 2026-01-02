# %% [markdown]
# # Group Analysis
# *Written by Luke Chang, updated for v0.6.0*
#
# In fMRI analysis, we are primarily interested in making inferences about how the brain processes information across all brains - not just the participants in our study. This requires estimating population-level effects from a sample of participants.
#
# In this tutorial, we will cover:
# - The two-level model (first-level GLM → second-level group analysis)
# - Running one-sample t-tests across subjects
# - Computing contrasts between conditions
# - Using design matrices for different statistical tests
#
# ## The Two-Level Model
#
# fMRI analysis typically uses a hierarchical approach:
#
# **First Level (Single Subject):**
# $$Y_i = X_i\beta + \epsilon_i$$
#
# where $\epsilon_i \sim \mathcal{N}(0, \sigma_i^2)$
#
# **Second Level (Group):**
# $$\beta = X_g\beta_g + \eta$$
#
# where $\eta \sim \mathcal{N}(0, \sigma_g^2)$
#
# The key insight is that the variance *within* a subject is usually smaller than variance *across* subjects. By modeling these separately, we get more accurate population-level estimates.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from nltools.datasets import fetch_haxby
from nltools.utils import concatenate

# %% [markdown]
# ## Loading the Haxby Dataset
#
# We'll use the Haxby dataset, which contains fMRI data from subjects viewing different categories of images (faces, houses, cats, etc.). This is a classic dataset for demonstrating group-level analyses.

# %%
# Load data for all subjects, then use first 3 for speed
# n_subjects='all' returns nested structure: subjects → runs → BrainData
print("Loading Haxby dataset...")
all_data, all_dm = fetch_haxby(n_subjects="all", verbose=1)

# Use first 3 subjects for this tutorial (faster than all 6)
n_subjects = 3
all_subjects = all_data[:n_subjects]
all_subjects_dm = all_dm[:n_subjects]

print(f"\nUsing {len(all_subjects)} subjects")
print(f"Subject 1 has {len(all_subjects[0])} runs")
print(f"Each run has {len(all_subjects[0][0])} timepoints")

# %% [markdown]
# ## First-Level Analysis: Fitting GLMs for Each Subject
#
# Before we can do group analysis, we need parameter estimates (betas) for each subject.
# We'll fit a GLM to each subject's first run and extract the contrast maps.

# %%
# Fit GLM for each subject and compute contrasts
contrasts_face = []
contrasts_house = []
contrasts_face_vs_house = []

for i, (subj_runs, subj_dm) in enumerate(zip(all_subjects, all_subjects_dm)):
    print(f"Processing subject {i + 1}...")

    # Use first run for speed
    data = subj_runs[0].copy()
    dm = subj_dm[0]

    # Add nuisance regressors (drift, polynomial)
    dm_filt = dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)

    # Fit the GLM
    data.fit(model="glm", X=dm_filt)

    # Compute contrasts for face and house (vs implicit baseline)
    contrasts_face.append(data.compute_contrasts("face"))
    contrasts_house.append(data.compute_contrasts("house"))

    # Compute face > house contrast
    contrasts_face_vs_house.append(data.compute_contrasts("face - house"))

print(f"\nExtracted contrasts for {len(contrasts_face)} subjects")

# %% [markdown]
# ## Second-Level Analysis: One-Sample T-Test
#
# Now we can test whether there is consistent activation to faces across subjects.
# The one-sample t-test asks: Is the mean activation significantly different from zero?
#
# $$H_0: \mu = 0 \quad \text{vs} \quad H_1: \mu \neq 0$$

# %%
# Stack face contrasts into a single BrainData object
face_group = concatenate(contrasts_face)
print(f"Group data shape: {face_group.shape} (subjects x voxels)")

# Run voxel-wise one-sample t-test using scipy
# This tests whether each voxel's mean across subjects differs from zero
t_values, p_values = ttest_1samp(face_group.data, 0, axis=0)
print(f"\nT-test computed for {len(t_values)} voxels")

# %% [markdown]
# ### Visualizing Group Results
#
# Let's look at the group-level activation to faces.

# %%
# Plot mean activation across subjects
print("Mean activation to faces across subjects:")
face_group.mean().plot(title="Mean Face Response")

# %%
# Plot t-statistic map
# Create a BrainData object from t-values for visualization
t_map_face = face_group[0].copy()  # Copy to get mask/affine
t_map_face.data = t_values.reshape(1, -1)  # Set data to t-values
print("\nT-statistic map (group activation to faces):")
t_map_face.plot(title="T-statistic: Face Response")

# %% [markdown]
# ## Contrast Analysis: Face vs House
#
# A more interesting question: Which regions respond *more* to faces than houses?
# This is a paired contrast, computed as: $\beta_{face} - \beta_{house}$
#
# We already computed this contrast at the first level. Now we test it at the group level.

# %%
# Stack the face vs house contrasts and run t-test
contrast_group = concatenate(contrasts_face_vs_house)
print(f"Contrast group shape: {contrast_group.shape}")

# Run voxel-wise one-sample t-test on the contrast
t_values_contrast, p_values_contrast = ttest_1samp(contrast_group.data, 0, axis=0)
print(f"T-test computed for {len(t_values_contrast)} voxels")

# %% [markdown]
# ### Visualizing the Contrast
#
# Positive values indicate regions more active for faces than houses.
# We expect to see activation in the fusiform face area (FFA).

# %%
# Plot the contrast t-statistic
# Create a BrainData object from t-values for visualization
t_map_contrast = contrast_group[0].copy()
t_map_contrast.data = t_values_contrast.reshape(1, -1)
print("T-statistic: Face > House")
t_map_contrast.plot(title="Face > House")

# %%
# We can also look at the mean contrast
print("\nMean contrast (Face - House):")
contrast_group.mean().plot(title="Mean (Face - House)")

# %% [markdown]
# ## Understanding Design Matrices for Group Analysis
#
# Different statistical tests are just special cases of the general linear model.
# The design matrix determines what hypothesis you're testing.
#
# | Test | Design Matrix |
# |------|---------------|
# | One-sample t-test | Vector of ones (intercept) |
# | Two-sample t-test | Group indicator + intercept |
# | Paired t-test | Condition contrast + subject means |
# | ANOVA | Multiple group indicators |
#
# ### Simulation: One-Sample T-Test as Regression
#
# The one-sample t-test is equivalent to fitting an intercept-only model.

# %%
# Simulate some "beta values" from subjects
np.random.seed(42)
simulated_betas = 5 + np.random.randn(20) * 2  # Mean=5, SD=2

# Method 1: Standard t-test
t_stat, p_value = ttest_1samp(simulated_betas, 0)
print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")

# Method 2: Regression with intercept only
X = np.ones((len(simulated_betas), 1))  # Design matrix: just an intercept
beta = np.linalg.lstsq(X, simulated_betas, rcond=None)[0]
print(f"\nRegression intercept (= mean): {beta[0]:.3f}")
print(f"Actual mean: {np.mean(simulated_betas):.3f}")

# %% [markdown]
# ### Simulation: Paired T-Test
#
# For paired data (same subjects, two conditions), we compute the difference
# and test whether it's different from zero.

# %%
# Simulate paired data
np.random.seed(42)
n_subjects = 15
condition_A = 10 + np.random.randn(n_subjects) * 3
condition_B = 8 + np.random.randn(n_subjects) * 3  # Slightly lower mean

# The paired t-test is just a one-sample t-test on the differences
differences = condition_A - condition_B
t_stat, p_value = ttest_1samp(differences, 0)

print(f"Mean difference (A - B): {np.mean(differences):.3f}")
print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(
    ["Condition A", "Condition B"],
    [np.mean(condition_A), np.mean(condition_B)],
    yerr=[
        np.std(condition_A) / np.sqrt(n_subjects),
        np.std(condition_B) / np.sqrt(n_subjects),
    ],
)
axes[0].set_ylabel("Mean Value (± SEM)")
axes[0].set_title("Condition Means")

axes[1].hist(differences, bins=10, edgecolor="black")
axes[1].axvline(0, color="red", linestyle="--", label="Null hypothesis")
axes[1].axvline(np.mean(differences), color="green", linestyle="-", label="Sample mean")
axes[1].set_xlabel("Difference (A - B)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Differences")
axes[1].legend()

plt.tight_layout()

# %% [markdown]
# ## Summary
#
# In this tutorial, we covered:
#
# 1. **Two-level modeling**: First-level GLM for each subject → Second-level group analysis
# 2. **One-sample t-test**: Testing whether mean activation differs from zero
# 3. **Contrast analysis**: Comparing conditions (e.g., Face > House)
# 4. **Design matrices**: Understanding that different tests are special cases of regression
#
# ### Key Methods Used
#
# | Method | Description |
# |--------|-------------|
# | `fetch_haxby()` | Load the Haxby dataset |
# | `BrainData.fit(model='glm')` | Fit first-level GLM |
# | `BrainData.compute_contrasts()` | Compute contrasts from fitted GLM |
# | `concatenate()` | Stack multiple BrainData objects |
# | `scipy.stats.ttest_1samp()` | Run group-level one-sample t-test |
#
# ### Next Steps
#
# - Tutorial 04: Thresholding group results (FDR, cluster correction)
# - Tutorial 05: Representational Similarity Analysis (RSA)
# - Tutorial 06: Multivariate Pattern Analysis (MVPA)
