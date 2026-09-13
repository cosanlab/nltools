"""The retained release imports independently of deferred collection support."""

import subprocess
import sys


def test_retained_release_imports_without_collection():
    script = """
import sys
import nltools
from nltools import algorithms, cross_validation, datasets, io, mask, models
from nltools.data import Adjacency, BrainData, DesignMatrix, Predict

assert 'nltools.data.collection' not in sys.modules
assert 'BrainCollection' not in nltools.data.__all__
assert 'PredictCollection' not in nltools.data.__all__
assert 'permutation_scores' not in Predict.__dataclass_fields__
assert 'permutation_pvalue' not in Predict.__dataclass_fields__
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
