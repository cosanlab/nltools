import pytest


@pytest.fixture(scope='function')
def sim():
    from nltools import simulator
    return simulator.Simulator()
