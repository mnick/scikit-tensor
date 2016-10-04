from numpy.random import randint, seed
import pytest


@pytest.fixture
def sptensor_seed():
    return seed(5)


@pytest.fixture
def sz():
    return 100


@pytest.fixture
def vals(sptensor_seed, sz):
    return randint(0, 100, sz)


@pytest.fixture
def shape():
    return (25, 11, 18, 7, 2)


@pytest.fixture
def subs(sptensor_seed, shape, sz):
    return tuple(randint(0, shape[i], sz) for i in range(len(shape)))
