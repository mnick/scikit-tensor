from numpy import array
import pytest


@pytest.fixture
def subs():
    return (
        array([0, 1, 0, 5, 7, 8]),
        array([2, 0, 4, 5, 3, 9]),
        array([0, 1, 2, 2, 1, 0])
    )


@pytest.fixture
def vals():
    return array([1, 2, 3, 4, 5, 6.1])


@pytest.fixture
def shape():
    return (10, 12, 3)
