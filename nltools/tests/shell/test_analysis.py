import pytest
from nltools.simulator import Simulator
from nltools.analysis import Roc


def test_roc(tmpdir):
    """Test that deprecated predict method raises NotImplementedError."""
    sim = Simulator()
    sigma = 0.1
    y = [0, 1]
    n_reps = 10
    dat = sim.create_data(y, sigma, reps=n_reps, output_dir=None)

    algorithm = "svm"
    extra = {"kernel": "linear"}

    # Test that predict raises NotImplementedError
    with pytest.raises(NotImplementedError) as excinfo:
        dat.predict(algorithm=algorithm, plot=False, **extra)

    assert "deprecated" in str(excinfo.value).lower()
    assert "Model class" in str(excinfo.value)

    # Note: The original ROC testing functionality will need to be
    # reimplemented using the Model class in Priority 3
