from nltools.data import Brain_Collection, Brain_Data


def test_brain_collection(sim_brain_data):
    # sim_brain_data is a Brain_Data of shape 8 x num_voxels, i.e 8 values per voxel

    # Setup data-structure that would be returned by Brain_Data.regress()
    result = {
        "z_score": sim_brain_data,
        "t": sim_brain_data.copy(),
        "p": sim_brain_data.copy(),
        "beta": sim_brain_data.copy(),
        "se": sim_brain_data.copy(),
        "rsquared": sim_brain_data.copy()[0],  # 1 value per voxel
        "labels": [  # Haxby, 2001
            "bottle",
            "cat",
            "chair",
            "face",
            "house",
            "scissors",
            "shoe",
            "scrambledpix",
        ],
        # ommitting residual and predicted
    }

    # SINGELTON TESTS
    # Create a Brain_Collection from the result
    collection = Brain_Collection(result)

    # Initializing with a dict means length is 1
    assert len(collection) == 1
    assert collection.keys() == sorted(result.keys())

    # All Labels, Single Key
    # Like a pandas dataframe we can access underlying Brain
    # implicitly the same as collection[:, "beta", :].shape()
    # Result 2d Brain_Data (8 x vox); all 8 category betas
    assert collection["beta"].shape() == sim_brain_data.shape()

    # Single Label, Single Key
    # With a regressors list we can also do label based indexing *into* each Brain_Data
    # Result: 1d Brain_Data (vox,)
    assert collection[:, "beta", "face"].shape() == sim_brain_data[0].shape()

    #
    assert collection[:, :, "face"].shape() == sim_brain_data[0].shape()

    # Multi-label, Single Key
    # TODO: currently returns dict, should return Brain_Data
    # And complicated 2d slicing
    # Result: 2d Brain_Data (2 x vox,); face_beta, house_beta
    # assert collection[["face", "house"], "beta"].shape() == sim_brain_data[:2].shape()

    # Multi-label, Multi-key
    # TODO:
    # write test

    # MULTIPLE ITEM TESTS

    # Brain_Collection is particulary useful for handling as *list* of dictionaries, i.e.
    # working with the results of multiple regressions
    collection = Brain_Collection([result, result, result, result])
    assert len(collection) == 4

    # Positional Indexing that makes a Brain_Collection a singleton with a single key
    # automatically flattens to a Brain_Data instance
    # Result 2d Brain_Data; all 8 category betas from 1st result
    out = collection[0, "beta"]
    assert out.shape() == sim_brain_data[0].shape()
    assert isinstance(out, Brain_Data)

    # Label-based indexing with a single key intelligently combines slices from
    # underlying dictionaries into a 2d Brain_Data of shape collection_len x vox
    out = collection["face", "beta"]
    assert out.shape()[0] == 4
    assert isinstance(out, Brain_Data)

    # TODO:
    # Multi-position, Single Key
    # Currently we return a list of dicts or list of Brain_Data; this need to be fixed
    # However if either the positional slice or number of keys is > 1 we always return a Brain_Collection
    # Result Brain_Collection of length 2 with a single key "beta"
    out = collection[:2, "beta"]
    assert out.shape() == (2, sim_brain_data[0].shape())
    assert isinstance(out, Brain_Collection)

    out = collection[0, ["beta", "t"]]
    assert isinstance(out, Brain_Collection)
    assert len(out) == 1
    assert out.keys() == ["beta", "t"]

    # TODO:
    # Multi-label, Single Key
    # Should return Brain_Collection of length = original length with 2 keys 'face', 'house'
    out = collection[["face", "house"], "beta"]
    assert isinstance(out, Brain_Collection)
    assert len(out) == 4
    assert out.keys() == ["face", "house"]

    # TODO:
    # Multi-label, Multi-key
    # Should return Brain_Collection of length = original length with 3 keys 'beta', 't', 'p'
    # and 2 regressors: "face", "house"
    # Each value is a 2d Brain_Data of shape 2 x vox; face_*, house_*
    out = collection[["face", "house"], ["beta", "t", "p"]]
    assert isinstance(out, Brain_Collection)

    # TODO:
    # Multi-position, Multi-key
    # Should return Brain_Collection of length = positional-slice 2 keys 'beta', 't'
    # Write test
    out = collection[:2, ["beta", "t"]]
