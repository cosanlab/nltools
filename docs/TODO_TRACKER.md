# Tutorial TODO Tracker for nltools v0.6.0

This document tracks which tutorials have commented-out code waiting for Priority 3 feature implementation.

## Tutorials Waiting for Model Class Implementation

### Waiting for Model.predict() Method
- [ ] `docs/tutorials/02_Analysis/plot_multivariate_classification.ipynb` - 6 `.predict()` calls commented
- [ ] `docs/tutorials/02_Analysis/plot_multivariate_prediction.ipynb` - 9 `.predict()` calls commented

### Waiting for Model.ttest() Method
- [ ] `docs/tutorials/02_Analysis/plot_univariate_regression.ipynb` - 2 `.ttest()` calls commented
- [ ] `docs/tutorials/basic/04_basic_analysis_workflow.ipynb` - 1 `.ttest()` call commented
- [ ] `docs/tutorials/basic/Haxby_2001.ipynb` - 1 `.ttest()` call commented

### Waiting for Brain_Collection Class
- [ ] `docs/tutorials/basic/05_brain_collection_basics.ipynb` - Entire tutorial commented (class not yet implemented)

## Fully Working Tutorials (v0.6.0)
These tutorials have no deprecated method calls and work with v0.6.0:
- [x] `docs/tutorials/basic/01_brain_data_basics.ipynb`
- [x] `docs/tutorials/basic/02_design_matrix_basics.ipynb`
- [x] `docs/tutorials/basic/03_adjacency_basics.ipynb`
- [x] `docs/tutorials/01_DataOperations/` - All tutorials in this directory
- [x] `docs/tutorials/02_Analysis/plot_decomposition.ipynb`
- [x] `docs/tutorials/02_Analysis/plot_similarity.ipynb`
- [x] `docs/tutorials/02_Analysis/plot_hyperalignment.ipynb`

## New Tutorials to Create (when Priority 3 is complete)
- [ ] `plot_compute_contrasts.ipynb` - Demonstrate new `.compute_contrasts()` method
- [ ] Update regression tutorials to show new `.regress()` attribute-based API

## Notes
- All commented code includes `# TODO: Update when Model.<method>() is implemented (Priority 3)` markers
- Documentation builds successfully with commented code (`jupyter-book build docs/`)
- Uncomment code blocks as Priority 3 features are implemented and tested

Last Updated: 2025-10-28