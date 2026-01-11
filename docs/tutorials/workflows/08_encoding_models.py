# %% [markdown]
# # Encoding Models: Predicting Brain from Stimulus
#
# This tutorial demonstrates nltools' encoding model workflows - predicting
# brain activity from stimulus features using GLM and Ridge regression.
#
# ## When to Use Which
#
# | Goal | Method | API | Output |
# |------|--------|-----|--------|
# | Which voxels respond to X? | GLM | `fit(model='glm')` | β, t, p per voxel |
# | How well can X predict brain? | Ridge | `fit(model='ridge', cv=5)` | R² per voxel |
#
# **GLM**: Parameter inference - get significance values for each regressor
#
# **Ridge**: Predictive performance - get cross-validated R² scores

# %%
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

from nltools.data import BrainCollection
from nltools.datasets import fetch_haxby

# %% [markdown]
# ## Load Data
#
# We'll use the Haxby dataset - participants viewed faces, houses, and objects.

# %%
# Load single subject
brain_data, design_matrices = fetch_haxby(n_subjects=1, verbose=0)
data = brain_data[0]
dm = design_matrices[0]

print(f"Brain data shape: {data.shape}")
print(f"Design matrix columns: {list(dm.columns)[:5]}...")

# %% [markdown]
# ## GLM Encoding: Parameter Inference
#
# Use GLM when you want to know **which voxels significantly respond** to
# your conditions. Returns β weights, t-statistics, and p-values.

# %%
# Fit GLM
data.fit(model="glm", X=dm)

# Access results
print("GLM results available:")
print(f"  - glm_betas: {data.glm_betas.shape}")
print(f"  - glm_t: {data.glm_t.shape}")
print(f"  - glm_p: {data.glm_p.shape}")

# %% [markdown]
# ### Compute Contrasts
#
# Test specific hypotheses by computing contrasts:

# %%
# Face vs house contrast
contrast = data.compute_contrasts("face - house")
print(f"Contrast shape: {contrast.shape}")

# Threshold and visualize
thresholded = contrast.threshold(lower="95%")
thresholded.plot(title="Face > House (top 5%)")

# %% [markdown]
# ## Ridge Encoding: Predictive Performance
#
# Use Ridge when you want to know **how well features predict brain activity**.
# Cross-validation gives honest out-of-sample R² scores.

# %%
# Prepare feature matrix (use condition columns as features)
feature_cols = ["face", "house", "cat", "bottle", "scissors", "shoe", "chair"]
X = dm[feature_cols].to_numpy()
print(f"Feature matrix shape: {X.shape}")

# Fit Ridge with cross-validation
data.fit(model="ridge", X=X, cv=5, alpha="auto")

# Access results
print("\nRidge results available:")
print(f"  - ridge_weights: {data.ridge_weights.shape}")
print(f"  - ridge_scores: {data.ridge_scores.shape}")
print(f"  - cv_results_['best_alpha']: {data.cv_results_['best_alpha']}")
print(f"  - cv_results_['mean_score']: {data.cv_results_['mean_score'].mean():.4f}")

# %%
# Visualize predictable regions
predictable = data.ridge_scores.threshold(lower="95%")
predictable.plot(title="Most Predictable Voxels (top 5% R²)")

# %% [markdown]
# ### Predict on New Data
#
# Use fitted model to generate predictions:

# %%
# Create new features (e.g., first 10 timepoints)
X_new = X[:10]

# Generate predictions
predictions = data.predict(X=X_new)
print(f"Predictions shape: {predictions.shape}")

# %% [markdown]
# ## Side-by-Side Comparison
#
# Both methods applied to same data reveal complementary information:

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# GLM: Which voxels respond to faces?
face_t = data.glm_t[dm.columns.index("face")]
face_t.plot(axes=axes[0], title="GLM: Face t-statistic")

# Ridge: Which voxels are predictable from all features?
data.ridge_scores.plot(axes=axes[1], title="Ridge: Prediction R²")

plt.tight_layout()

# %% [markdown]
# **Key difference**:
# - GLM tells you where effects are *significant* (p-values)
# - Ridge tells you where effects are *predictable* (R²)

# %% [markdown]
# ## Multi-Subject Encoding with BrainCollection
#
# For group-level encoding analysis, use `BrainCollection`:

# %%
# Load multiple subjects (fetch_haxby returns runs for one subject at a time)
# Here we load subjects 1, 2, 3 and take their first run each
subjects_data = []
subjects_dm = []
for subj_id in [1, 2, 3]:
    runs, dms = fetch_haxby(n_subjects=subj_id, verbose=0)
    subjects_data.append(runs[0])  # First run
    subjects_dm.append(dms[0])

mask = subjects_data[0].mask
bc = BrainCollection(subjects_data, mask=mask)

print(f"BrainCollection shape: {bc.shape}")
print(f"Number of subjects: {bc.n_images}")

# %% [markdown]
# ### Fit Ridge Across Subjects

# %%
# Prepare features (same for all subjects in this example)
n_timepoints = bc[0].shape[0]
X_group = subjects_dm[0][feature_cols].to_numpy()

# Fit ridge to each subject, get both scores and weights
result = bc.fit_ridge(X=X_group, cv=3, output="both", show_progress=False)

print(f"Result keys: {list(result.keys())}")
print(f"Weights shape: {result['weights'].shape}")
print(f"Scores shape: {result['scores'].shape}")

# %% [markdown]
# ### Group-Level Weight Inference
#
# Test if weights are consistent across subjects:

# %%
# Get weights for all subjects
weights = result["weights"]  # BrainCollection of weights

# Mean weights across subjects
mean_weights = weights.mean(axis=0)
print(f"Mean weights shape: {mean_weights.shape}")

# Test if face weights are significantly different from zero
# (weights[:, face_idx, :] for the face feature)
face_idx = feature_cols.index("face")

# Get face weights for all subjects - each subject has (n_features, n_voxels)
# So we need to extract face weights across subjects
face_weights_list = []
for subj_weights in weights:
    face_weights_list.append(subj_weights[face_idx])

# Stack into BrainCollection for group test
face_weights_bc = BrainCollection(face_weights_list, mask=mask)
t_stat, p_val = face_weights_bc.ttest()

print(f"Significant face encoding voxels: {(p_val.data < 0.05).sum()}")

# %% [markdown]
# ## ROI-Level Encoding
#
# For hypothesis-driven analysis, extract ROI timeseries and model:

# %%
# Create a functional ROI from significant face voxels
face_roi = data.glm_t[dm.columns.index("face")].threshold(lower="95%", binarize=True)
print(f"ROI size: {face_roi.data.sum():.0f} voxels")

# Extract mean ROI timeseries
roi_timeseries = data.extract_roi(face_roi)
print(f"ROI timeseries shape: {roi_timeseries.shape}")

# %% [markdown]
# ### Model ROI Activity

# %%
# Predict ROI activity from features
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
scores = cross_val_score(ridge, X, roi_timeseries, cv=5)

print(f"ROI prediction R²: {scores.mean():.3f} ± {scores.std():.3f}")

# Fit final model to see feature importance
ridge.fit(X, roi_timeseries)
print("\nFeature weights for ROI:")
for feat, weight in zip(feature_cols, ridge.coef_):
    print(f"  {feat}: {weight:.4f}")

# %% [markdown]
# ## Summary
#
# | Workflow | API | Use Case |
# |----------|-----|----------|
# | GLM inference | `brain.fit(model='glm', X=dm)` | Significance testing |
# | Ridge prediction | `brain.fit(model='ridge', X=X, cv=5)` | Predictive performance |
# | Contrasts | `brain.compute_contrasts("A - B")` | Hypothesis testing |
# | Multi-subject | `bc.fit_ridge(X=X)` | Group encoding |
# | ROI encoding | `extract_roi()` + sklearn | Targeted analysis |
#
# All workflows return results as BrainData objects for easy visualization
# and further analysis.
