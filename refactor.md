# Brain_Data

## Goal
Reduce total class code by being more efficient about using existing functionality from dependecy libraries, especially nilearn which has included many new features since we first wrote `nltools`. We want to "reduce our code surface" so that future maintenance becomes dramatically easier. 

## Methods and attributes to remove
- `.predict()` -> remove
- `.randomise()` -> remove
- `.ttest()` -> remove
- `.predict()` -> remove
- `.predict_multi()` -> remove
- `.X` -> `.design_matrix` only when `.regress(design_matrix)` is called
- `.Y` -> remove

### New Methods to add
- `.compute_contrasts()` method that takes strings (referring to column names of the `.design_matrix` attribute) or numpy array to calculate contrasts as linear recombinations of .`glm_betas` by performing multiplation between the `.glm_betas` and the contrast vector. Should use `nilearn` compute_contrasts funcionality (which already handles string based contrasts and comparisons) and save result to `.glm_contrast` attribute

### Methods to refactor
- `__init__()` 
  - should handle a new optional masker kwarg that enables the use of alternative nilearn maskers in particular the non-Multi variants of LabelsMasker, MapsMasker, and SpheresMasker
  - just like current functionality converts input nifti/h5 file into a `.data` attribute containg observations x voxels, these alternative maskers should convert `.data` to observationx x labels/masks/spheres, etc 
  - all subsequent methods should "just work" but now operating over labels/masks instead of voxels
- `.apply_mask()`
  - should use nilearn functionality if existing or clean up into simpler numpy ops with no copying if no functionality exists
- `.extract_roi()`
  - should now use [`nilearn.maskers.NiftiLabelsMasker`](https://nilearn.github.io/stable/modules/generated/nilearn.maskers.NiftiLabelsMasker.html#nilearn.maskers.NiftiLabelsMasker) instead to transform data into observations x labels representation just like we currently using `NiftiMasker` to load data into observations x voxels representation
- `.regress()`
  - should require as input a `Design_Matrix` or pandas DataFrame which gets saved to `.design_matrix` attribute
  - should no longer return a dictionary or `Brain_Collection` but store results as `Brain_Data` attributes (e.g. `self.glm_betas`, `self.glm_t`, `self.glm_p`, etc) 




