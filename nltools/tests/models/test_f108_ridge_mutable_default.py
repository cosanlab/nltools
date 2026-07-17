"""F108: Ridge.concentration must not share a mutable default across instances."""

from nltools.models import Ridge


def test_concentration_default_not_shared():
    """Two default-constructed Ridge models must hold independent concentration lists."""
    m1 = Ridge()
    m2 = Ridge()
    assert m1.concentration == [0.1, 1.0]
    assert m1.concentration is not m2.concentration
    m1.concentration.append(999.0)
    assert 999.0 not in m2.concentration
