# %% [markdown]
# # Thresholding Group Analyses
#
# *Written by Luke Chang, updated for v0.6.0*
#
# Now that we have learned how to estimate a single-subject model, create contrasts, and run a group-level analysis, the next important topic to cover is how we can threshold these group maps. This is not as straightforward as it might seem as we need to be able to correct for multiple comparisons.
#
# In this tutorial, we will cover:
#
# - Issues with correcting for multiple comparisons
# - Family Wise Error Rate (FWER)
# - Bonferroni Correction
# - False Discovery Rate (FDR)
# - Applying thresholds to real brain data
#
# ## Understanding Multiple Comparisons
#
# The primary goal in fMRI data analysis is to make inferences about how the brain processes information. These inferences can be in the form of predictions, but most often we are testing hypotheses about whether a particular region of the brain is involved in a specific type of process.
#
# Hypothesis testing in fMRI is complicated by the fact that we are running many tests across each voxel in the brain (hundreds of thousands of tests). Selecting an appropriate threshold requires finding a balance between sensitivity (true positive rate) and specificity (false negative rate).
#
# **Type I error**: $H_0$ is true, but we mistakenly reject it (False Positive) - controlled by $\alpha$
#
# **Type II error**: $H_0$ is false, but we fail to reject it (False Negative)
#
# There are two main approaches to correcting for multiple tests:
#
# 1. **Familywise Error Rate (FWER)**: Control the probability of finding *any* false positives
# 2. **False Discovery Rate (FDR)**: Control the *proportion* of false positives among significant tests

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from nltools.simulator import SimulateGrid
from nltools.stats import fdr
from nltools.datasets import fetch_haxby
from nltools.utils import concatenate

# %% [markdown]
# ## Simulations: Understanding False Positives
#
# Let's explore the concept of false positives using simulations to build intuition
# about the challenges of multiple comparisons correction.
#
# We'll generate 100 x 100 voxels from a standard normal distribution for 20 independent participants.

# %%
simulation = SimulateGrid(grid_width=100, n_subjects=20)

f, a = plt.subplots(nrows=5, ncols=4, figsize=(15, 15), sharex=True, sharey=True)
counter = 0
for col in range(4):
    for row in range(5):
        sns.heatmap(
            simulation.data[:, :, counter],
            ax=a[row, col],
            cmap="RdBu_r",
            vmin=-4,
            vmax=4,
        )
        a[row, col].set_title(f"Subject {counter + 1}", fontsize=16)
        counter += 1
plt.tight_layout()

# %% [markdown]
# Each subject's simulated data is on a 100 x 100 grid. Think of this as a slice from their brain.
# We have generated random noise separately for each subject - no true signal yet.
#
# Now let's run an independent one-sample t-test on every pixel across all 20 participants.

# %%
simulation.fit()

sns.heatmap(simulation.t_values, square=True, cmap="RdBu_r", vmin=-4, vmax=4)
plt.title("T Values", fontsize=18)

# %% [markdown]
# Even though there was no signal, you can see pixels that exceed t > 2 (approximately p < 0.05).
# These are all **false positives**.
#
# ### Effect of Threshold on False Positives
#
# Let's run 100 simulations to estimate the false positive rate at different thresholds.

# %%
threshold_value = 0.05
simulation = SimulateGrid(grid_width=100, n_subjects=20)
simulation.plot_grid_simulation(
    threshold=threshold_value, threshold_type="p", n_simulations=100
)

# %% [markdown]
# With p < 0.05 and 10,000 voxels, we observe false positives in almost every simulation!
#
# What if we use a much more stringent threshold?

# %%
threshold_value = 0.0001
simulation = SimulateGrid(grid_width=100, n_subjects=20)
simulation.plot_grid_simulation(
    threshold=threshold_value, threshold_type="p", n_simulations=100
)

# %% [markdown]
# This dramatically decreases false positives. Some simulations now have zero false positives.
#
# ## Bonferroni Correction
#
# The Bonferroni correction divides alpha by the number of tests:
#
# $$\alpha_{bonf} = \frac{\alpha}{M}$$
#
# where $M$ is the number of voxels.

# %%
grid_width = 100
bonf_threshold = 0.05 / (grid_width**2)
print(f"Bonferroni threshold: p < {bonf_threshold:.2e}")

simulation = SimulateGrid(grid_width=grid_width, n_subjects=20)
simulation.plot_grid_simulation(
    threshold=bonf_threshold, threshold_type="p", n_simulations=100
)

# %% [markdown]
# Bonferroni controls the false positive rate, but what happens when there's real signal?
#
# Let's add a 10x10 square of true signal in the middle of our grid.

# %%
grid_width = 100
bonf_threshold = 0.05 / (grid_width**2)
signal_amplitude = 1

simulation = SimulateGrid(
    signal_amplitude=signal_amplitude,
    signal_width=10,
    grid_width=grid_width,
    n_subjects=20,
)
simulation.plot_grid_simulation(
    threshold=bonf_threshold, threshold_type="p", n_simulations=100
)

# %% [markdown]
# The false positive rate is controlled, but we're only recovering about 12% of the true signal!
#
# This highlights the main issue with Bonferroni: it's so conservative that we miss real effects.
#
# ## False Discovery Rate (FDR)
#
# FDR controls the *proportion* of false positives among significant tests, rather than
# controlling *any* false positive. This is often more appropriate for fMRI.
#
# The FDR procedure:
# 1. Select a desired limit $q$ on FDR (e.g., 0.05)
# 2. Rank all p-values from smallest to largest
# 3. Find threshold $r$ such that $p \leq \frac{i}{m} \times q$
# 4. Reject any $H_0$ with p-value below $r$

# %%
grid_width = 100
signal_amplitude = 1

simulation = SimulateGrid(
    signal_amplitude=signal_amplitude,
    signal_width=10,
    grid_width=grid_width,
    n_subjects=20,
)
simulation.plot_grid_simulation(
    threshold=0.05, threshold_type="q", n_simulations=100, correction="fdr"
)
print(
    f"FDR q < 0.05 corresponds to p-value threshold of {simulation.corrected_threshold:.6f}"
)

# %% [markdown]
# FDR recovers much more of the true signal while maintaining reasonable control over false discoveries.
#
# Let's examine the distribution of false discovery rates across simulations:

# %%
plt.hist(simulation.multiple_fdr, bins=20, edgecolor="black")
plt.ylabel("Frequency", fontsize=14)
plt.xlabel("False Discovery Rate", fontsize=14)
plt.axvline(0.05, color="red", linestyle="--", label="q = 0.05")
plt.legend()
plt.title("FDR Distribution Across Simulations", fontsize=14)

# %% [markdown]
# Most simulations have a false discovery rate below our target q < 0.05.
#
# ## Thresholding Real Brain Data
#
# Now let's apply these concepts to real fMRI data from the Haxby dataset.
# We'll compute group statistics for face vs house activation.

# %%
# Load Haxby dataset (3 subjects for speed)
print("Loading Haxby dataset...")
all_data, all_dm = fetch_haxby(n_subjects="all", verbose=1)

n_subjects = 3
all_subjects = all_data[:n_subjects]
all_subjects_dm = all_dm[:n_subjects]

print(f"\nUsing {len(all_subjects)} subjects")

# %%
# Fit GLM for each subject and compute face vs house contrast
contrasts_face_vs_house = []

for i, (subj_runs, subj_dm) in enumerate(zip(all_subjects, all_subjects_dm)):
    print(f"Processing subject {i + 1}...")

    # Use first run for speed
    data = subj_runs[0].copy()
    dm = subj_dm[0]

    # Add nuisance regressors
    dm_filt = dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)

    # Fit the GLM
    data.fit(model="glm", X=dm_filt)

    # Compute face > house contrast
    contrasts_face_vs_house.append(data.compute_contrasts("face - house"))

print(f"\nExtracted contrasts for {len(contrasts_face_vs_house)} subjects")

# %%
# Run group-level t-test
contrast_group = concatenate(contrasts_face_vs_house)
print(f"Group data shape: {contrast_group.shape}")

# Voxel-wise one-sample t-test
t_values, p_values = ttest_1samp(contrast_group.data, 0, axis=0)
print(f"Number of voxels: {len(t_values)}")

# %% [markdown]
# ### Uncorrected Threshold
#
# First, let's look at results with an uncorrected threshold of p < 0.001.
# This is arbitrary and does NOT control for multiple comparisons.

# %%
# Create BrainData objects for visualization
t_map = contrast_group[0].copy()
t_map.data = t_values.reshape(1, -1)

p_map = contrast_group[0].copy()
p_map.data = p_values.reshape(1, -1)

# Threshold at p < 0.001 uncorrected
unc_threshold = 0.001
sig_mask_unc = p_values < unc_threshold
n_sig_unc = np.sum(sig_mask_unc)
print(f"Uncorrected (p < {unc_threshold}): {n_sig_unc} significant voxels")

# Create thresholded map
t_map_unc = t_map.copy()
t_map_unc.data[0, ~sig_mask_unc] = 0

print("\nFace > House (uncorrected p < 0.001):")
t_map_unc.plot(title="Face > House (uncorrected)")

# %% [markdown]
# ### Bonferroni Correction
#
# Now let's apply Bonferroni correction.

# %%
n_voxels = len(p_values)
bonf_threshold = 0.05 / n_voxels
print(f"Bonferroni threshold: p < {bonf_threshold:.2e}")

sig_mask_bonf = p_values < bonf_threshold
n_sig_bonf = np.sum(sig_mask_bonf)
print(f"Bonferroni corrected: {n_sig_bonf} significant voxels")

# Create thresholded map
t_map_bonf = t_map.copy()
t_map_bonf.data[0, ~sig_mask_bonf] = 0

print("\nFace > House (Bonferroni corrected):")
if n_sig_bonf > 0:
    t_map_bonf.plot(title="Face > House (Bonferroni)")
else:
    print("No voxels survive Bonferroni correction")

# %% [markdown]
# With only 3 subjects, the Bonferroni threshold is very stringent and may yield no results.
# This demonstrates the power issue with small samples.
#
# ### FDR Correction
#
# Let's try FDR correction, which is more sensitive.

# %%
# Compute FDR threshold
fdr_threshold = fdr(p_values, q=0.05)
print(f"FDR threshold (q < 0.05): p < {fdr_threshold:.6f}")

if fdr_threshold > 0:
    sig_mask_fdr = p_values < fdr_threshold
    n_sig_fdr = np.sum(sig_mask_fdr)
    print(f"FDR corrected: {n_sig_fdr} significant voxels")

    # Create thresholded map
    t_map_fdr = t_map.copy()
    t_map_fdr.data[0, ~sig_mask_fdr] = 0

    print("\nFace > House (FDR q < 0.05):")
    t_map_fdr.plot(title="Face > House (FDR corrected)")
else:
    print("No voxels survive FDR correction")

# %% [markdown]
# ### Comparing Thresholds
#
# Let's summarize the results from different thresholding approaches:

# %%
print("Thresholding Summary:")
print("-" * 50)
print(f"Total voxels tested: {n_voxels:,}")
print(
    f"Uncorrected (p < 0.001): {n_sig_unc:,} voxels ({100 * n_sig_unc / n_voxels:.2f}%)"
)
print(f"Bonferroni (p < {bonf_threshold:.2e}): {n_sig_bonf:,} voxels")
if fdr_threshold > 0:
    print(f"FDR (q < 0.05, p < {fdr_threshold:.6f}): {n_sig_fdr:,} voxels")
else:
    print("FDR (q < 0.05): 0 voxels (no threshold found)")

# %% [markdown]
# ## Summary
#
# In this tutorial, we covered:
#
# 1. **Multiple comparisons problem**: Testing thousands of voxels inflates false positives
# 2. **FWER (Bonferroni)**: Controls any false positive, but very conservative
# 3. **FDR**: Controls proportion of false positives, more sensitive
# 4. **Practical application**: Applied thresholds to real Haxby data
#
# ### Key Points
#
# | Method | Controls | Power | Use Case |
# |--------|----------|-------|----------|
# | Uncorrected | Nothing | High | Exploratory only |
# | Bonferroni | Any FP | Low | Very conservative |
# | FDR | Proportion of FPs | Medium | Standard approach |
#
# ### Key Functions Used
#
# | Function | Description |
# |----------|-------------|
# | `SimulateGrid` | Simulate grid data for multiple comparisons |
# | `scipy.stats.ttest_1samp()` | One-sample t-test |
# | `nltools.stats.fdr()` | Compute FDR threshold |
# | `nltools.stats.threshold()` | Threshold stat map by p-values |
#
# ### Next Steps
#
# - Tutorial 05: Representational Similarity Analysis (RSA)
# - Tutorial 06: Multivariate Pattern Analysis (MVPA)
