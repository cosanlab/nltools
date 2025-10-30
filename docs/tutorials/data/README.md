# Tutorial Data

This directory contains lightweight test data used across nltools tutorials.

## Data Files

Currently, all tutorials use `nltools.datasets.fetch_*()` functions to download data on-demand. This keeps the repository lean while providing realistic examples.

### Available Datasets

**Via `fetch_pain()`**:
- Pain perception study data
- 84 subjects × 238,955 voxels (MNI 2mm space)
- Downloaded to `~/.nltools/data/` on first use
- Used in: Brain_Data basics, GLM tutorials, Group analysis

**Via `fetch_localizer()`**:
- Event-related localizer data
- Used in: Advanced tutorials

## Adding Custom Data

To add small test datasets (<5 MB) to this directory:

1. Ensure data is de-identified and shareable
2. Document source and preprocessing
3. Update this README with description
4. Reference in relevant tutorials

## Data Organization

```
docs/tutorials/data/
├── README.md                    # This file
├── example_events.csv           # (Planned) Example event file
└── minimal_brain_data.h5        # (Planned) Minimal test data
```

**Note**: Currently planned but not yet implemented. All tutorials use fetched data.
