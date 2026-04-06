---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
execute:
  skip: true
---

# Multivariate Pattern Analysis (MVPA)

Classification using face vs. house decoding from the Haxby dataset.

```{code-cell} python3
import matplotlib

matplotlib.use("Agg")

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from nltools.datasets import fetch_haxby
```

## Load Data

```{code-cell} python3
print("Loading Haxby dataset...")
all_data, all_dm = fetch_haxby(n_subjects="all", verbose=1)

n_subjects = 3
print(f"\nUsing {n_subjects} subjects")
```

## Extract Face and House Samples

```{code-cell} python3
X_list = []
y_list = []
groups_list = []

for subj_idx in range(n_subjects):
    data = all_data[subj_idx][0]
    dm = all_dm[subj_idx][0]

    face_mask = dm["face"].to_numpy() > 0.5
    house_mask = dm["house"].to_numpy() > 0.5

    face_data = data[face_mask].data
    house_data = data[house_mask].data

    X_list.append(face_data)
    X_list.append(house_data)
    y_list.extend([1] * len(face_data))
    y_list.extend([0] * len(house_data))
    groups_list.extend([subj_idx] * (len(face_data) + len(house_data)))

    print(f"Subject {subj_idx + 1}: {len(face_data)} face, {len(house_data)} house")

X = np.vstack(X_list)
y = np.array(y_list)
groups = np.array(groups_list)

print(f"\nTotal: {len(y)} samples, {X.shape[1]} voxels")
```

## Train Classifier

```{code-cell} python3
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = SVC(kernel="linear", C=1.0)
clf.fit(X_scaled, y)

train_accuracy = clf.score(X_scaled, y)
print(f"Training accuracy: {train_accuracy * 100:.1f}%")
```

## Cross-Validation (Leave-One-Subject-Out)

```{code-cell} python3
logo = LeaveOneGroupOut()
cv_scores = cross_val_score(clf, X_scaled, y, cv=logo, groups=groups)

print("LOSO cross-validation:")
for i, score in enumerate(cv_scores):
    print(f"  Fold {i + 1}: {score * 100:.1f}%")
print(f"\nMean: {cv_scores.mean() * 100:.1f}% (±{cv_scores.std() * 100:.1f}%)")
print("Chance: 50%")
```

## Classifier Weights

```{code-cell} python3
weights = clf.coef_[0]

template = all_data[0][0][0]
weight_brain = template.copy()
weight_brain.data = weights.reshape(1, -1)

print(f"Positive (face): {(weights > 0).sum()} voxels")
print(f"Negative (house): {(weights < 0).sum()} voxels")

weight_brain.plot(title="SVM Weights: Face (+) vs House (-)")
```

## Feature Selection

```{code-cell} python3
# Keep top 10% variance voxels
voxel_var = np.var(X, axis=0)
threshold = np.percentile(voxel_var, 90)
selected_voxels = voxel_var >= threshold

print(f"Selected: {selected_voxels.sum()} of {len(voxel_var)} voxels")

X_selected = X[:, selected_voxels]
X_selected_scaled = scaler.fit_transform(X_selected)

cv_scores_selected = cross_val_score(clf, X_selected_scaled, y, cv=logo, groups=groups)

print(f"CV accuracy: {cv_scores_selected.mean() * 100:.1f}%")
print(f"Full brain: {cv_scores.mean() * 100:.1f}%")
```

## Regularization

```{code-cell} python3
c_values = [0.01, 0.1, 1.0, 10.0]

print("Effect of C parameter:")
for c in c_values:
    clf_reg = SVC(kernel="linear", C=c)
    scores = cross_val_score(clf_reg, X_scaled, y, cv=logo, groups=groups)
    print(f"  C={c:5.2f}: {scores.mean() * 100:.1f}%")
```

## Summary

```{code-cell} python3
print("=" * 40)
print("MVPA: Face vs House")
print("=" * 40)
print(f"Subjects: {n_subjects}")
print(f"Samples: {(y == 1).sum()} face, {(y == 0).sum()} house")
print(f"Voxels: {X.shape[1]}")
print(f"\nLOSO CV: {cv_scores.mean() * 100:.1f}%")
print("Chance: 50%")
```
