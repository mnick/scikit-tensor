from ..utils import accum
from numpy import array, allclose


def test_accum():
    subs1 = array([0, 1, 1, 2, 2, 2])
    subs2 = array([0, 1, 1, 1, 2, 2])
    vals = array([1, 2, 3, 4, 5, 6])
    nvals, nsubs = accum((subs1, subs2), vals, with_subs=True)
    assert allclose(nvals, array([1, 5, 4, 11]))
    assert allclose(nsubs[0], array([0, 1, 2, 2]))
    assert allclose(nsubs[1], array([0, 1, 1, 2]))

    subs1 = array([0, 0, 1])
    subs2 = array([0, 0, 1])
    vals = array([1, 2, 3])
    nvals, nsubs = accum((subs1, subs2), vals, with_subs=True)
    assert allclose(nvals, array([3, 3]))
    assert allclose(nsubs[0], array([0, 1]))
    assert allclose(nsubs[1], array([0, 1]))
