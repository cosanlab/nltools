from nltools.data import Brain_Collection, Brain_Data


def test_collection_basics(sim_brain_data):
    # Setup
    # sim_brain_data is a Brain_Data fixture with shape determined by the test data
    # 8 x num_vox (~240k)
    # Check the actual shape to set up labels correctly
    n = sim_brain_data.shape()[0]

    # Overview
    # Brain_Collection is a collection of Brain_Data objects
    # that's internally represented as a list of dictionaries who's values
    # are 1d or 2d Brain_Data objects

    # Check valid initializations
    # 1. Empty
    col = Brain_Collection()
    assert len(col) == 0
    assert col.labels is None
    assert col.keys() == []

    # Otherwise col.data is always internally represented a **list of dictionaries**

    # 2. Single dict; -> internally list of length 1
    col = Brain_Collection({"beta": sim_brain_data})
    assert len(col) == 1
    assert col.labels is None
    assert col.keys() == ["beta"]

    # 3. List of dicts; no labels
    col = Brain_Collection(
        [
            {
                "beta": sim_brain_data,
            },
            {
                "beta": sim_brain_data,
            },
        ]
    )
    assert len(col) == 2
    assert col.labels is None
    assert col.keys() == ["beta"]

    # 4. List of dicts; with labels
    # 'labels' is a special optional dictionary key which must be the same on all collection items if provided
    # if provided it's extracted to the col.labels attribute and popped out of each dictionary, thus never returned as part of slicing operations
    # when provided it supports label-based indexing along the last dimension of the collection (see below)
    labels = [f"condition_{i}" for i in range(n)]

    col_with_labels = Brain_Collection(
        [
            {
                "beta": sim_brain_data,
                "t": sim_brain_data,
                "labels": labels,
            },
            {
                "beta": sim_brain_data,
                "t": sim_brain_data,
                "labels": labels,
            },
        ]
    )
    assert len(col_with_labels) == 2
    assert col_with_labels.labels == labels
    assert col_with_labels.keys() == ["beta", "t"]

    # Test slicing
    # col[position, key, label]

    # position(s) = which list item(s) of the internal data structure
    # key(s) = which key(s) from individual dict items
    # label(s) = how to slice individual Brain_Data objects

    # position(s) supports numerical index, slicing, and list-of-int based indexing similar to 1d numpy arrays
    # key(s) only supports string/name based indexing or list-of-strings based indexing similar to slicing multiple polars dataframe columns
    # label(s) by default behabes like position(s), but if labels attribute exists, also supporst behaving like key(s)

    # Slicing always tries to *intelligently flatten* the result
    # to a 2d Brain_Data if any 2 out of 3 slicing dimensions are singletons:

    # single position + single key = 2d Brain_Data with .shape()[0] == len(labels)
    # single position + single label = 2d Brain_Data with .shape()[0] == len(keys)
    # single key + single label = 2d Brain_Data with .shape()[0] == len(collection)

    # otherwise it will return a Brain_Collection with strictly the same or fewer
    # numbers of positions, keys, and/or lanels

    # We can always flatten a single position + single key to a Brain_Data
    assert isinstance(col[0, "beta"], Brain_Data)
    # This is equivalent to:
    assert isinstance(col[0, "beta", :], Brain_Data)

    # Slicing along the *last* dimension determines the shape of Brain_Data
    assert len(col[0, "beta", 0].shape()) == 1  # 1D
    assert col[0, "beta", :3].shape() == (
        3,
        sim_brain_data.shape()[1],
    )  # 2D with 3 x voxels

    # Slicing by position will always return a Brain_Collection
    assert isinstance(col[:, "beta"], Brain_Collection)

    # like before, indexing/slicing along the *last* dimension determines the shape
    # of each Brain_Data inside of the collection
    s = col[:, "beta", :]  # explicit version of previous
    assert isinstance(s, Brain_Collection)
    assert len(s) == len(col)
    assert s[0, "beta"].shape()[0] == n
    assert s[0, "beta"].shape()[1] == sim_brain_data.shape()[1]

    # More explicit verison
    assert col[:, "beta", :] == col[:, "beta"]

    # Since we have no labels we can only use integer-based index for the label dimension
    assert col[0, "beta", 0] == col[0, "beta"][0]

    # When we select a single label and a single key, we intelligently concatenate slices
    # aross different Brain_Data objects, e.g. the *first* beta for each item in col
    assert isinstance(col[:, "beta", 0], Brain_Data)
    assert col[:, "beta", 0].shape()[0] == len(col.data)

    # When we have labels we can also do this as:
    assert col_with_labels[:, "beta", "condition_0"] == col[:, "beta", 0]

    # More complex slicing operation but result should be clear = collection because we have more than 1 key or label
    s = col_with_labels[0, ["beta", "t"], ["condition_0", "condition_1"]]
    assert isinstance(s, Brain_Collection)
    assert len(s) == 1
    assert s.keys() == ["beta", "t"]
    assert s.labels == ["condition_0", "condition_1"]
    b = s[0, "beta", "condition_0"]
    assert isinstance(b, Brain_Data)
    assert len(b.shape()) == 1  # 1D array when slicing with single index
    assert b.shape()[0] == sim_brain_data.shape()[1]  # num_voxels


def test_regress_collection(sim_brain_data):
    # Setup data-structure that would be returned by Brain_Data.regress()
    # Create labels based on actual data shape
    n_conditions = sim_brain_data.shape()[0]
    labels = ["condition_" + str(i) for i in range(n_conditions)]

    # Make face and house special indices for testing
    if n_conditions >= 4:
        labels[3] = "face"
    if n_conditions >= 5:
        labels[4] = "house"

    result = {
        "z_score": sim_brain_data,
        "t": sim_brain_data.copy(),
        "p": sim_brain_data.copy(),
        "beta": sim_brain_data.copy(),
        "se": sim_brain_data.copy(),
        "rsquared": sim_brain_data.copy()[0],  # 1 value per voxel
        "labels": labels,
        # ommitting residual and predicted
    }

    # TODO: write these
    # SINGLETON TESTS
    # Create a Brain_Collection from the result
    singleton = Brain_Collection(result)

    # TODO: expand these
    # Test appending
    singleton.append(result)
    assert len(singleton) == 2
    assert set(singleton.keys()) == {"z_score", "t", "p", "beta", "se", "rsquared"}
    assert singleton.labels == labels

    # TODO: write these
    # MULTIPLE TESTS
    collection = Brain_Collection([result, result, result, result])
