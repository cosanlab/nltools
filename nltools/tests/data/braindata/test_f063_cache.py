"""Regression test for F063: CacheManager.load used allow_pickle=True.

The cache stores only plain numeric arrays, but ``load`` passed
``allow_pickle=True``, turning any tampered/corrupt cache file into an
arbitrary-code-execution vector. Loading an object-array member should now
raise (pickle disabled) rather than silently unpickling it.
"""

import numpy as np
import pytest

from nltools.data.braindata.cache import CacheManager


class TestCacheNoPickle:
    def test_load_numeric_roundtrip(self, tmp_path):
        """Normal numeric arrays still round-trip through save/load."""
        cm = CacheManager()
        cm.cache_dir = tmp_path
        cm.save("mykey", indices=np.arange(5), shape=np.array([2, 3]))
        loaded = cm.load("mykey")
        np.testing.assert_array_equal(loaded["indices"], np.arange(5))
        np.testing.assert_array_equal(loaded["shape"], np.array([2, 3]))

    def test_load_rejects_object_array(self, tmp_path):
        """An object-array member must not be unpickled on load."""
        cm = CacheManager()
        cm.cache_dir = tmp_path
        path = cm.get_path("tampered")
        # Craft a cache file containing a pickled object array.
        np.savez(path, payload=np.array([{"evil": 1}], dtype=object))

        with pytest.raises(ValueError):
            # dict(np.load(...)) materializes every member; with allow_pickle
            # False this raises on the object array instead of unpickling it.
            cm.load("tampered")
