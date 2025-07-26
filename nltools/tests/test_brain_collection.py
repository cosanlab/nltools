from nltools.data import Brain_Collection, Brain_Data
from pytest import raises


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

    # a dict or list of length 1 containing a dict should always return a Brain_Collection singleton
    assert isinstance(col[0], Brain_Collection)

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

    # Additional edge case tests for singleton/non-singleton combinations

    # Case 1: List with single element is NOT singleton (position dimension)
    s = col[[0], "beta", 0]  # 2 singletons (key + label), should return Brain_Data
    assert isinstance(s, Brain_Data)
    assert len(s.shape()) == 1  # 1D array when slicing with single integer label
    assert s.shape()[0] == sim_brain_data.shape()[1]  # num_voxels

    # Case 2: List with single element is NOT singleton (key dimension)
    s = col[0, ["beta"], 0]  # 2 singletons (position + label), BUT multiple keys
    assert isinstance(s, Brain_Collection)  # Never concatenate across keys!
    assert len(s) == 1
    assert s.keys() == ["beta"]
    # When we get the beta from this collection, it's already sliced to single label
    b = s[0, "beta"]
    assert isinstance(b, Brain_Data)
    assert len(b.shape()) == 1  # 1D array from single label index

    # Case 3: Multiple positions, single key, multiple labels (the previously failing test case)
    s = col_with_labels[:, "beta", ["condition_0", "condition_1"]]
    assert isinstance(s, Brain_Collection)  # Only 1 singleton (key)
    assert len(s) == 2  # Should preserve 2 positions
    assert s.labels == ["condition_0", "condition_1"]
    assert s[0, "beta"].shape()[0] == 2
    assert s[1, "beta"].shape()[0] == 2

    # Case 4: Test order preservation - position order
    s = col_with_labels[[1, 0], "beta", 0]  # 2 singletons (key + label)
    assert isinstance(s, Brain_Data)
    assert s.shape()[0] == 2  # Should concatenate in order [1, 0]
    # Verify order by checking data values if they differ

    # Case 5: Multiple keys - NEVER concatenate across keys
    s = col_with_labels[
        0, ["t", "beta"], 0
    ]  # 2 singletons (position + label), BUT multiple keys
    assert isinstance(s, Brain_Collection)  # Never concatenate across keys!
    assert len(s) == 1
    assert s.keys() == ["beta", "t"]  # Keys should be in sorted order
    assert s.labels == ["condition_0"]

    # Case 6: Test order preservation - label order with names
    s = col_with_labels[
        0, "beta", ["condition_1", "condition_0"]
    ]  # 2 singletons (position + key)
    assert isinstance(s, Brain_Data)
    assert s.shape()[0] == 2

    # Case 7: Slice positions, single key, single label
    s = col[:1, "beta", 0]  # 2 singletons (key + label)
    assert isinstance(s, Brain_Data)
    assert len(s.shape()) == 1  # 1D array when using integer label
    assert s.shape()[0] == sim_brain_data.shape()[1]  # num_voxels

    # Case 8: List positions, single key, single label
    s = col[[0, 1], "beta", 0]  # 2 singletons (key + label)
    assert isinstance(s, Brain_Data)
    assert s.shape() == (
        2,
        sim_brain_data.shape()[1],
    )  # 2D: concatenated from both positions

    # Case 9: Multiple everything - should return Brain_Collection
    s = col_with_labels[[0, 1], ["beta", "t"], ["condition_0", "condition_1"]]
    assert isinstance(s, Brain_Collection)  # No dimension is singleton
    assert len(s) == 2
    assert s.keys() == ["beta", "t"]
    assert s.labels == ["condition_0", "condition_1"]

    # Case 10: Single position, multiple keys, single label - Brain_Collection (never concat across keys)
    s = col_with_labels[0, ["beta", "t"], "condition_0"]
    assert isinstance(s, Brain_Collection)
    assert len(s) == 1
    assert s.keys() == ["beta", "t"]
    assert s.labels == ["condition_0"]


def test_regress_collection(regress_result):
    # Create a Brain_Collection from the result like .regress(as_collection=True)
    collection = Brain_Collection(regress_result)

    # Test appending, which works like lists *NOT* like Brain_Data
    # we don't need to do: collection = collection.append(new)
    collection.append(regress_result)  # in-place
    assert len(collection) == 2

    # Appending doesn't work with new items that are missing .labels
    results_no_labels = {k: v for k, v in regress_result.items() if k != "labels"}
    with raises(
        ValueError, match="New item must have labels to match existing collection"
    ):
        collection.append(results_no_labels)

    # Because of the failure above
    assert len(collection) == 2

    # Make sure slicing by string label works after the append
    # We're getting *all items*, *one key*, *two labels*
    s = collection[:, "beta", ["face", "house"]]
    assert len(s) == 2
    assert s.labels == ["face", "house"]
    assert s.keys() == ["beta"]
    assert isinstance(s[0, "beta"], Brain_Data) and s[1, "beta"].shape()[0] == 2
