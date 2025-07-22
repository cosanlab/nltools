from nltools.data import Brain_Collection, Brain_Data


def test_brain_collection(sim_brain_data):
    # sim_brain_data is a Brain_Data fixture with shape determined by the test data
    # Check the actual shape to set up labels correctly
    n_conditions = sim_brain_data.shape()[0]

    # Setup data-structure that would be returned by Brain_Data.regress()
    # Create labels based on actual data shape
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

    # SINGLETON TESTS
    # Create a Brain_Collection from the result
    collection = Brain_Collection(result)

    # Initializing with a dict means length is 1
    assert len(collection) == 1
    # Note: 'labels' should not be in keys as it's extracted
    assert 'labels' not in collection.keys()
    assert collection.labels == result["labels"]

    # All Items, All Labels, Single Key
    # Like a pandas dataframe we can access underlying Brain
    # implicitly the same as collection[:, "beta", :].shape()
    # Result 2d Brain_Data (8 x vox); all 8 category betas
    assert collection["beta"].shape() == sim_brain_data.shape()

    # All Items, Single Key, Single Label
    # Result: 2d Brain_Data (len_collection, vox)
    assert collection[:, "beta", "face"].shape() == sim_brain_data[0].shape()

    # All Items, All Keys, Single Label
    # Result: Brain_Collection singleton with all keys each of which contain
    # Brain_Data sliced by label
    face_all_keys = collection[:, :, "face"]
    assert isinstance(face_all_keys, Brain_Collection)
    assert len(face_all_keys) == 1
    # Each key should have the face data
    assert face_all_keys["beta"].shape() == sim_brain_data[3].shape()  # face is index 3

    # All Items, Single Key, Multi-labels
    # Result: Brain_Collection with 2 labels worth of data
    face_house = collection[:, "beta", ["face", "house"]]
    assert isinstance(face_house, Brain_Data)
    # Should have 2 rows (face and house) x num_voxels
    assert face_house.shape()[0] == 2

    # Single Item, Single Key, Single Label
    # Result: 1d Brain_Data (voxels only)
    single_result = collection[0, "beta", "face"]
    assert isinstance(single_result, Brain_Data)
    assert single_result.shape() == sim_brain_data[3].shape()  # face is index 3

    # MULTIPLE ITEM TESTS

    # Brain_Collection is particularly useful for handling a *list* of dictionaries, i.e.
    # working with the results of multiple regressions
    # Create a copy of result without modifying the original
    result_copy = result.copy()
    collection = Brain_Collection([result_copy, result_copy, result_copy, result_copy])
    assert len(collection) == 4
    assert collection.labels == result["labels"]

    # Positional Indexing: single item, single key
    # Result: Brain_Data with all labels from 1st result
    out = collection[0, "beta"]
    assert out.shape() == sim_brain_data.shape()
    assert isinstance(out, Brain_Data)

    # All items, single key, single label
    # Result: 2d Brain_Data of shape (n_items x voxels)
    out = collection[:, "beta", "face"]
    assert out.shape()[0] == 4  # 4 items
    assert isinstance(out, Brain_Data)

    # Multi-position, Single Key
    # Result: Brain_Collection of length 2 with a single key "beta"
    out = collection[:2, "beta"]
    assert isinstance(out, Brain_Collection)
    assert len(out) == 2
    assert out.keys() == ["beta"]

    # Single position, Multi-key
    # Result: Brain_Collection with multiple keys
    out = collection[0, ["beta", "t"]]
    assert isinstance(out, Brain_Collection)
    assert len(out) == 1
    assert sorted(out.keys()) == ["beta", "t"]

    # Multi-label, Single Key
    # All items, single key, multiple labels
    # Result: concatenated Brain_Data with 2 labels per item
    out = collection[:, "beta", ["face", "house"]]
    assert isinstance(out, Brain_Data)
    # Should have n_items * n_labels rows
    assert out.shape()[0] == 8  # 4 items * 2 labels

    # Multi-label, Multi-key
    # Result: Brain_Collection with multiple keys, each containing multi-label data
    out = collection[:, ["beta", "t", "p"], ["face", "house"]]
    assert isinstance(out, Brain_Collection)
    assert len(out) == 4  # Same number of items
    assert sorted(out.keys()) == ["beta", "p", "t"]
    # Each key should contain data for 2 labels
    assert out[0, "beta"].shape()[0] == 2

    # Multi-position, Multi-key
    # Result: Brain_Collection of length 2 with 2 keys
    out = collection[:2, ["beta", "t"]]
    assert isinstance(out, Brain_Collection)
    assert len(out) == 2
    assert sorted(out.keys()) == ["beta", "t"]
    
    # EDGE CASES
    
    # Test empty collection
    empty = Brain_Collection()
    assert len(empty) == 0
    assert empty.labels is None
    assert empty.keys() == []
    
    # Test setting labels after initialization
    collection_no_labels = Brain_Collection([{
        "beta": sim_brain_data,
        "t": sim_brain_data.copy()
    }])
    assert collection_no_labels.labels is None
    
    # Set labels
    collection_no_labels.set_labels(result["labels"], for_keys="beta")
    assert collection_no_labels.labels == result["labels"]
    
    # Test error when accessing non-existent label
    try:
        collection[:, "beta", "nonexistent"]
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not found" in str(e)
    
    # Test error when no labels but trying label indexing
    try:
        collection_no_labels_2 = Brain_Collection([{"beta": sim_brain_data}])
        collection_no_labels_2[:, "beta", "face"]
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No labels found" in str(e)
