# %% [markdown]
# # Multivariate Pattern Analysis (MVPA)
#
# *Written by Luke Chang, updated for v0.6.0*
#
# This tutorial introduces multivariate pattern analysis (MVPA) and classification
# using the classic Haxby face vs. house decoding example.
#
# In contrast to univariate analyses that test each voxel independently, MVPA
# identifies **patterns** of activity across many voxels that predict psychological states.
#
# $$\text{outcome} = \sum_{i}^n \beta_i \cdot \text{voxel}_i + \epsilon$$
#
# We will cover:
# 1. **Data preparation** - extracting labeled samples for classification
# 2. **Training classifiers** - using scikit-learn's SVM
# 3. **Cross-validation** - estimating generalization performance
# 4. **Model interpretation** - understanding classifier weights

# %%
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from nltools.datasets import fetch_haxby

# %% [markdown]
# ## Why MVPA?
#
# Univariate GLM analyses localize which regions are associated with specific
# processes by testing each voxel independently. MVPA instead identifies
# **distributed representations** - patterns across many voxels that encode
# information about psychological states.
#
# Key insight: A single voxel might show no difference between faces and houses,
# but the *pattern* of activity across thousands of voxels can reliably
# discriminate between them.

# %% [markdown]
# ## Data Preparation
#
# We'll use the Haxby dataset to classify face vs. house viewing.
# This is a classic MVPA demonstration showing that visual cortex contains
# distributed representations of object categories.

# %%
print("Loading Haxby dataset...")
all_data, all_dm = fetch_haxby(n_subjects="all", verbose=1)

# Use first 3 subjects for this tutorial
n_subjects = 3
print(f"\nUsing {n_subjects} subjects for classification")

# %% [markdown]
# ### Extracting Face and House Timepoints
#
# For classification, we need labeled examples. We'll extract the raw
# timepoints when participants were viewing faces vs. houses.

# %%
# Collect data from all subjects
X_list = []  # Brain data
y_list = []  # Labels
groups_list = []  # Subject IDs for cross-validation

for subj_idx in range(n_subjects):
    # Use first run from each subject
    data = all_data[subj_idx][0]
    dm = all_dm[subj_idx][0]

    # Get timepoints for face and house conditions
    face_mask = dm["face"].to_numpy() > 0.5
    house_mask = dm["house"].to_numpy() > 0.5

    # Extract brain data for each condition
    face_data = data[face_mask].data
    house_data = data[house_mask].data

    # Combine and create labels
    X_list.append(face_data)
    X_list.append(house_data)
    y_list.extend([1] * len(face_data))  # 1 = face
    y_list.extend([0] * len(house_data))  # 0 = house
    groups_list.extend([subj_idx] * (len(face_data) + len(house_data)))

    print(f"Subject {subj_idx + 1}: {len(face_data)} face, {len(house_data)} house")

# Stack all data
X = np.vstack(X_list)
y = np.array(y_list)
groups = np.array(groups_list)

print(f"\nTotal: {len(y)} samples ({(y == 1).sum()} face, {(y == 0).sum()} house)")
print(f"Features: {X.shape[1]} voxels")

# %% [markdown]
# ## Training a Classifier
#
# We'll use a Support Vector Machine (SVM) with a linear kernel.
# SVMs find a hyperplane in feature (voxel) space that maximally
# separates the two classes.
#
# Important preprocessing: We standardize the features (z-score each voxel)
# so that all voxels contribute equally regardless of their baseline intensity.

# %%
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train SVM
clf = SVC(kernel="linear", C=1.0)
clf.fit(X_scaled, y)

# Training accuracy (will be optimistic due to overfitting)
train_accuracy = clf.score(X_scaled, y)
print(f"Training accuracy: {train_accuracy * 100:.1f}%")

# %% [markdown]
# The training accuracy is high, but this is **misleading** - the model may be
# overfitting to the training data. We need cross-validation for unbiased accuracy.

# %% [markdown]
# ## Cross-Validation
#
# To get an unbiased estimate of classifier performance, we use
# **leave-one-subject-out (LOSO) cross-validation**. This ensures:
#
# 1. Training and test data come from different subjects
# 2. We avoid inflated accuracy from temporal autocorrelation
# 3. We estimate how well the model generalizes to new participants

# %%
# Leave-one-subject-out cross-validation
logo = LeaveOneGroupOut()

cv_scores = cross_val_score(clf, X_scaled, y, cv=logo, groups=groups)

print("Leave-one-subject-out cross-validation:")
for i, score in enumerate(cv_scores):
    print(f"  Fold {i + 1} (test=subject {i + 1}): {score * 100:.1f}%")
print(
    f"\nMean CV accuracy: {cv_scores.mean() * 100:.1f}% (±{cv_scores.std() * 100:.1f}%)"
)
print("Chance level: 50%")

# %% [markdown]
# ### Interpreting Results
#
# An accuracy significantly above 50% (chance) indicates the model learned
# meaningful patterns that generalize across subjects.
#
# With real fMRI data, we typically expect 60-90% accuracy for face vs. house
# classification in ventral temporal cortex.

# %%
# Compare training vs cross-validated accuracy
print("\nAccuracy Comparison:")
print(f"  Training:        {train_accuracy * 100:.1f}%")
print(f"  Cross-validated: {cv_scores.mean() * 100:.1f}%")
print(f"  Difference:      {(train_accuracy - cv_scores.mean()) * 100:.1f}%")

if train_accuracy - cv_scores.mean() > 0.15:
    print("\n  Note: Gap suggests some overfitting to training data")

# %% [markdown]
# ## Visualizing the Classifier Weights
#
# The SVM learns a weight for each voxel. Positive weights indicate voxels
# that contribute to predicting "face", negative for "house".

# %%
# Get classifier weights
weights = clf.coef_[0]

# Create a BrainData object for visualization
# We need the mask from the original data
template = all_data[0][0][0]  # Single timepoint as template
weight_brain = template.copy()
weight_brain.data = weights.reshape(1, -1)

print("Classifier weight map:")
print(f"  Positive weights (→ face): {(weights > 0).sum()} voxels")
print(f"  Negative weights (→ house): {(weights < 0).sum()} voxels")
print(f"  Max weight: {weights.max():.4f}")
print(f"  Min weight: {weights.min():.4f}")

# Plot the weight map
weight_brain.plot(title="SVM Weights: Face (+) vs House (-)")

# %% [markdown]
# ## Feature Selection
#
# Using all ~40,000 voxels can lead to overfitting. Feature selection
# reduces dimensionality by keeping only informative voxels.
#
# Common approaches:
# - **Anatomical ROI**: Use a predefined brain region (e.g., fusiform gyrus)
# - **Variance filtering**: Keep high-variance voxels
# - **Univariate selection**: Keep voxels with significant condition differences

# %%
# Simple feature selection: keep top 10% most variable voxels
voxel_var = np.var(X, axis=0)
threshold = np.percentile(voxel_var, 90)
selected_voxels = voxel_var >= threshold

print(f"Feature selection: {selected_voxels.sum()} of {len(voxel_var)} voxels")

# Apply selection
X_selected = X[:, selected_voxels]
X_selected_scaled = scaler.fit_transform(X_selected)

# Cross-validate with selected features
cv_scores_selected = cross_val_score(clf, X_selected_scaled, y, cv=logo, groups=groups)

print("\nWith feature selection:")
print(
    f"  CV accuracy: {cv_scores_selected.mean() * 100:.1f}% (±{cv_scores_selected.std() * 100:.1f}%)"
)
print(f"  vs full brain: {cv_scores.mean() * 100:.1f}%")

# %% [markdown]
# ## Regularization
#
# Regularization prevents overfitting by penalizing model complexity.
# The C parameter in SVM controls regularization (smaller C = more regularization).

# %%
# Compare different regularization strengths
c_values = [0.01, 0.1, 1.0, 10.0]

print("Effect of regularization (C parameter):")
for c in c_values:
    clf_reg = SVC(kernel="linear", C=c)
    scores = cross_val_score(clf_reg, X_scaled, y, cv=logo, groups=groups)
    print(f"  C={c:5.2f}: {scores.mean() * 100:.1f}% (±{scores.std() * 100:.1f}%)")

# %% [markdown]
# ## Summary

# %%
print("=" * 50)
print("MVPA Classification Results: Face vs House")
print("=" * 50)
print(f"\nDataset: Haxby ({n_subjects} subjects)")
print(f"Samples: {(y == 1).sum()} face, {(y == 0).sum()} house")
print(f"Features: {X.shape[1]} voxels")
print("\nCross-validated accuracy (LOSO):")
print(f"  Full brain: {cv_scores.mean() * 100:.1f}% (±{cv_scores.std() * 100:.1f}%)")
print(f"  Top 10% variance: {cv_scores_selected.mean() * 100:.1f}%")
print("\nChance level: 50%")

# %% [markdown]
# ## Summary
#
# In this tutorial, we covered:
#
# 1. **Data preparation**: Extracting labeled samples from the Haxby dataset
# 2. **SVM classification**: Training a linear SVM with scikit-learn
# 3. **Cross-validation**: Using leave-one-subject-out for unbiased accuracy
# 4. **Weight visualization**: Understanding what the classifier learned
# 5. **Feature selection**: Reducing dimensions to improve generalization
# 6. **Regularization**: Controlling model complexity
#
# ### Key Concepts
#
# | Concept | Description |
# |---------|-------------|
# | MVPA | Using patterns across voxels to decode mental states |
# | Overfitting | Model learns noise, fails to generalize |
# | Cross-validation | Test on held-out data for unbiased accuracy |
# | LOSO CV | Leave-one-subject-out ensures independent test data |
# | Regularization | Penalize complexity to prevent overfitting |
#
# ### Further Reading
#
# - [Haxby et al., 2001](https://science.sciencemag.org/content/293/5539/2425) - Original face/house decoding
# - [Haynes & Rees, 2006](https://www.nature.com/articles/nrn1931) - MVPA review
# - [Pereira et al., 2009](https://doi.org/10.1016/j.neuroimage.2008.11.007) - Machine learning for neuroimaging
