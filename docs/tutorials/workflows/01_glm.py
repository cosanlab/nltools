# %% [markdown]
# # General Linear Model (GLM) Analysis
# *Written by Luke Chang*
#
# The General Linear Model (GLM) is the foundation of most fMRI data analysis. This tutorial will guide you through the complete workflow of analyzing fMRI data using GLM, from building design matrices to computing contrasts and performing group-level analyses.
#
# In this tutorial, we will cover:
# - Understanding the GLM framework
# - Building design matrices from experimental events
# - Adding nuisance variables (motion, drift, etc.)
# - Fitting GLM models to fMRI data
# - Computing contrasts to test hypotheses
# - Group-level analysis and thresholding
#
# ## Dataset
# We will work with the **Haxby dataset**, a classic fMRI dataset where participants viewed images of faces, houses, and other objects. This dataset is publicly available and perfect for learning GLM analysis.

# %%
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from nltools.datasets import fetch_haxby
from nltools.algorithms import glover_hrf

# %% [markdown]
# ## Loading the Haxby Dataset
#
# Let's start by loading the Haxby dataset. The `fetch_haxby()` function downloads and loads the data for us, returning both the fMRI data and design matrices.

# %%
# Load Haxby dataset for subject 1 (returns list of runs)
brain_data, design_matrices = fetch_haxby(n_subjects=1, verbose=1)

print(f"Loaded {len(brain_data)} runs")
print(f"First run has {len(brain_data[0])} timepoints")
print(f"Design matrix shape: {design_matrices[0].shape}")

# Use the first run for this tutorial
data = brain_data[0]
dm = design_matrices[0]

# %% [markdown]
# Let's examine the design matrix to see what conditions we have:

# %%
print("Design matrix columns:")
print(list(dm.columns))
print("\nDesign matrix shape:", dm.shape)
print("\nFirst few rows (showing first 3 regressors):")
# Convert to pandas for display
dm_pd = dm._to_pandas()
print(dm_pd[list(dm.columns[:3])].head())

# %% [markdown]
# ## Understanding the GLM
#
# The General Linear Model is a framework for modeling brain responses as a linear combination of predictors:
#
# $$Y = X\beta + \epsilon$$
#
# where:
# - $Y$ is the observed brain signal (voxel time series)
# - $X$ is the design matrix (predictors)
# - $\beta$ are the regression coefficients we want to estimate
# - $\epsilon$ is the error term
#
# Each column in the design matrix represents a different predictor (e.g., a condition or nuisance variable), and each row represents a timepoint.

# %% [markdown]
# ### Visualizing the Design Matrix
#
# Let's visualize the design matrix to understand its structure:

# %%
# Plot the design matrix as a heatmap
dm.heatmap(figsize=(12, 8))
plt.title("Design Matrix Heatmap", fontsize=16)

# %% [markdown]
# We can also plot individual regressors to see their time courses:

# %%
# Plot a few example regressors
dm_pd = dm._to_pandas()
fig, axes = plt.subplots(3, 1, figsize=(15, 9))
dm_pd[["face", "house"]].plot(ax=axes[0])
axes[0].set_title("Face and House Regressors", fontsize=14)
axes[0].legend()

dm_pd[["scrambledpix", "shoe"]].plot(ax=axes[1])
axes[1].set_title("Scrambled Pixels and Shoe Regressors", fontsize=14)
axes[1].legend()

# Plot all regressors together
dm_pd.plot(ax=axes[2], figsize=(15, 4))
axes[2].set_title("All Regressors", fontsize=14)
plt.tight_layout()

# %% [markdown]
# ## HRF Convolution
#
# Notice that the regressors in the design matrix are already convolved with the Hemodynamic Response Function (HRF). The HRF models the delayed and smoothed blood flow response to neural activity. This convolution is essential because the BOLD signal doesn't respond instantaneously to neural activity.
#
# Let's visualize what the HRF looks like:

# %%
# Generate HRF function
TR = 2.5  # Haxby dataset TR
hrf = glover_hrf(TR, oversampling=1)

plt.figure(figsize=(10, 4))
plt.plot(hrf, linewidth=3)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.title("Hemodynamic Response Function (HRF)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# %% [markdown]
# The HRF has a characteristic shape: a rapid rise, peak around 5-6 seconds, and a slower return to baseline with a slight undershoot.

# %% [markdown]
# ## Adding Nuisance Variables
#
# In addition to modeling our experimental conditions, we often need to account for various sources of noise and artifacts. Common nuisance variables include:
#
# 1. **High-pass filtering** (removing low-frequency drift)
# 2. **Motion parameters** (head movement)
# 3. **Spikes** (outlier timepoints)
# 4. **Intercepts** (baseline signal)
#
# ### High-Pass Filtering with DCT Basis
#
# fMRI data often contains slow drifts over time. We can model these using a Discrete Cosine Transform (DCT) basis set, which acts as a high-pass filter:

# %%
# Add DCT basis set (high-pass filter with 128s cutoff)
dm_filt = dm.add_dct_basis(duration=128)

print(f"Original design matrix shape: {dm.shape}")
print(f"After adding DCT basis: {dm_filt.shape}")
print("\nDCT regressors:")
print([col for col in dm_filt.columns if "dct" in col.lower()][:5])

# %% [markdown]
# Let's visualize the DCT basis functions:

# %%
# Plot DCT basis functions
dct_cols = [col for col in dm_filt.columns if "dct" in col.lower()]
if dct_cols:
    dm_filt_pd = dm_filt._to_pandas()
    fig, ax = plt.subplots(figsize=(15, 4))
    dm_filt_pd[dct_cols[:5]].plot(ax=ax)
    ax.set_title("DCT Basis Functions (High-Pass Filter)", fontsize=14)
    ax.legend()
    plt.tight_layout()

# %% [markdown]
# ### Adding Polynomial Trends
#
# We can also add polynomial trends (linear, quadratic, etc.) to model slow drifts. This is often used as an alternative or complement to DCT filtering:

# %%
# Add polynomial trends (up to order 2: intercept, linear, quadratic)
dm_filt_poly = dm_filt.add_poly(order=2, include_lower=True)

print(f"After adding polynomials: {dm_filt_poly.shape}")
print("\nPolynomial regressors:")
print([col for col in dm_filt_poly.columns if "poly" in col.lower()])

# %% [markdown]
# ### Checking for Multicollinearity
#
# When building design matrices, it's important to check for multicollinearity (high correlation between regressors), which can make coefficient estimates unstable. We can use the Variance Inflation Factor (VIF) to assess this:

# %%
# Calculate VIF for each regressor
vif_values = dm_filt_poly.vif()

# Plot VIF values
plt.figure(figsize=(12, 6))
plt.plot(range(len(vif_values)), vif_values, "o-", linewidth=2, markersize=8)
plt.axhline(y=10, color="r", linestyle="--", label="VIF = 10 (problematic)")
plt.axhline(y=4, color="orange", linestyle="--", label="VIF = 4 (investigate)")
plt.xlabel("Regressor Index", fontsize=12)
plt.ylabel("Variance Inflation Factor", fontsize=12)
plt.title("VIF for Design Matrix Regressors", fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# %% [markdown]
# VIF values greater than 10 indicate problematic multicollinearity. Values between 4-10 should be investigated. For our design matrix, the VIF values look reasonable.

# %% [markdown]
# ## Fitting the GLM Model
#
# Now that we have our design matrix ready, we can fit the GLM model to the fMRI data. The `fit()` method estimates the regression coefficients ($\beta$) for each voxel:

# %%
# Fit GLM model
data.fit(model="glm", X=dm_filt_poly)

print("GLM fitting complete!")
print(f"Number of beta maps: {len(data.glm_betas)}")
print(f"Beta maps shape: {data.glm_betas.shape}")

# %% [markdown]
# After fitting, the results are stored in several attributes:
# - `glm_betas`: Beta coefficients for each regressor
# - `glm_t`: T-statistics for each regressor
# - `glm_p`: P-values for each regressor
# - `glm_se`: Standard errors for each regressor
# - `glm_residuals`: Residuals from the model fit

# %% [markdown]
# Let's examine the beta maps for a few conditions:

# %%
# Get the index of the 'face' regressor
regressor_names = list(dm_filt_poly.columns)
face_idx = regressor_names.index("face") if "face" in regressor_names else None

if face_idx is not None:
    # Plot beta map for face condition
    face_beta = data.glm_betas[face_idx]
    face_beta.plot(title="Beta Map: Face Condition")

# %% [markdown]
# ## Computing Contrasts
#
# Contrasts allow us to test specific hypotheses by creating linear combinations of the beta coefficients. For example, we might want to test:
# - Which regions respond more to faces than houses?
# - Which regions respond to any visual stimulus?
#
# Contrasts can be specified in several ways:
# 1. **Numeric vectors**: `[0, 1, -1, 0, ...]` (weights for each regressor)
# 2. **String expressions**: `"face - house"` (more readable)
# 3. **Dictionaries**: Multiple contrasts at once

# %% [markdown]
# ### Example 1: Face vs House Contrast
#
# Let's compute a contrast comparing face responses to house responses:

# %%
# Method 1: Using string notation (most readable)
face_vs_house = data.compute_contrasts("face - house")
face_vs_house.plot(title="Face > House Contrast (T-statistic)")

# %% [markdown]
# ### Example 2: Average of Multiple Conditions
#
# We can also compute contrasts that average multiple conditions. For this, we'll use numeric contrast vectors:

# %%
# Average response to all visual conditions
visual_conditions = [
    "face",
    "house",
    "scrambledpix",
    "shoe",
    "bottle",
    "cat",
    "chair",
    "scissors",
]
# Create numeric contrast vector (equal weights for all visual conditions)
regressor_names = list(dm_filt_poly.columns)
contrast_vec = np.zeros(len(regressor_names))
for cond in visual_conditions:
    if cond in regressor_names:
        contrast_vec[regressor_names.index(cond)] = 1.0 / len(visual_conditions)

# Compute contrast
all_visual = data.compute_contrasts(contrast_vec)
all_visual.plot(title="Average Response to All Visual Conditions")

# %% [markdown]
# ### Example 3: Multiple Contrasts at Once
#
# We can compute multiple contrasts simultaneously using a dictionary:

# %%
# Define multiple contrasts
contrasts = {
    "face_vs_house": "face - house",
    "face_vs_scrambled": "face - scrambledpix",
    "faces_only": "face",
}

# For the objects contrast, use numeric vector
regressor_names = list(dm_filt_poly.columns)
objects_contrast = np.zeros(len(regressor_names))
for cond in ["shoe", "bottle", "chair", "scissors"]:
    if cond in regressor_names:
        objects_contrast[regressor_names.index(cond)] = 0.25  # Average of 4 conditions
contrasts["objects"] = objects_contrast

# Compute all contrasts
contrast_results = data.compute_contrasts(contrasts)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for idx, (name, result) in enumerate(contrast_results.items()):
    row = idx // 2
    col = idx % 2
    result.plot(ax=axes[row, col], title=name)
plt.tight_layout()

# %% [markdown]
# ## Examining Model Fit
#
# We can examine how well our model fits the data by looking at the residuals and model fit statistics:

# %%
# Plot residuals (should be normally distributed with no patterns)
residuals = data.glm_residual
mean_residuals = residuals.mean()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot residual map
mean_residuals.plot(ax=axes[0], title="Mean Residuals Across Brain")

# Plot histogram of residuals
axes[1].hist(mean_residuals.data.flatten(), bins=50, edgecolor="black")
axes[1].set_xlabel("Residual Value", fontsize=12)
axes[1].set_ylabel("Frequency", fontsize=12)
axes[1].set_title("Distribution of Residuals", fontsize=14)
axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
plt.tight_layout()

# %% [markdown]
# The residuals should be centered around zero and normally distributed. If there are patterns in the residuals, it suggests the model might be missing important predictors.

# %% [markdown]
# ## Group-Level Analysis
#
# So far, we've analyzed data from a single subject. To make inferences about the population, we need to perform group-level analyses. This typically involves:
#
# 1. **First-level analysis**: Fit GLM for each subject (what we just did)
# 2. **Second-level analysis**: Combine results across subjects
#
# Let's demonstrate this with multiple subjects:

# %%
# Load data for multiple subjects
all_subjects_data, all_subjects_dm = fetch_haxby(n_subjects="all", verbose=1)

print(f"Loaded {len(all_subjects_data)} subjects")
print(f"Subject 1 has {len(all_subjects_data[0])} runs")

# %% [markdown]
# For group analysis, we would:
# 1. Fit GLM for each subject (first-level)
# 2. Extract contrast maps for each subject
# 3. Perform a one-sample t-test across subjects (second-level)
#
# Here's a simplified example:

# %%
# Example: Extract face > house contrast for first subject, first run
# In practice, you'd do this for all subjects and runs
subject1_run1 = all_subjects_data[0][0]
subject1_dm = all_subjects_dm[0][0]

# Add nuisance variables
subject1_dm_filt = subject1_dm.add_dct_basis(duration=128).add_poly(
    order=2, include_lower=True
)

# Fit GLM
subject1_run1.fit(model="glm", X=subject1_dm_filt)

# Compute contrast
face_vs_house_sub1 = subject1_run1.compute_contrasts("face - house")

print("Contrast computed for subject 1, run 1")
face_vs_house_sub1.plot(title="Subject 1: Face > House")

# %% [markdown]
# ## Thresholding and Multiple Comparisons
#
# When examining contrast maps, we need to account for multiple comparisons (testing thousands of voxels). Common approaches include:
#
# 1. **Uncorrected thresholding**: Simple p-value threshold (e.g., p < 0.001)
# 2. **False Discovery Rate (FDR)**: Controls the expected proportion of false positives
# 3. **Family-Wise Error Rate (FWER)**: Controls the probability of any false positives
#
# The `BrainData` class provides methods for thresholding:

# %%
# Example: Threshold the contrast map
# Note: This is a simplified example. In practice, you'd use proper statistical thresholding

# Get t-statistic map (compute_contrasts returns t-statistics by default)
t_map = face_vs_house

# Simple thresholding (uncorrected)
# In practice, you'd use proper FDR or cluster correction
thresholded = t_map.copy()
thresholded.data[np.abs(thresholded.data) < 2.0] = 0  # Threshold at |t| > 2

thresholded.plot(title="Thresholded Contrast Map (|t| > 2)")

# %% [markdown]
# ## Summary
#
# In this tutorial, we've covered the complete GLM analysis workflow:
#
# 1. **Loading data**: Using `fetch_haxby()` to get fMRI data and design matrices
# 2. **Design matrices**: Understanding and building design matrices with experimental conditions
# 3. **Nuisance variables**: Adding DCT basis functions and polynomial trends to model drift
# 4. **Model fitting**: Using `fit(model="glm", X=design_matrix)` to estimate coefficients
# 5. **Contrasts**: Using `compute_contrasts()` to test specific hypotheses
# 6. **Group analysis**: Combining results across subjects
# 7. **Thresholding**: Accounting for multiple comparisons
#
# The GLM framework is powerful and flexible, allowing you to test a wide variety of hypotheses about brain function. For more advanced topics, see our other tutorials on decoding, RSA, and encoding models.
