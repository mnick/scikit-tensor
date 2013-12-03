from ..utils import accum
from nose.tools import assert_true
from numpy import array, allclose


def test_accum():
    subs1 = array([0, 1, 1])
    subs2 = array([0, 1, 1])
    vals = array([1, 2, 3])
    nvals, nsubs = accum((subs1, subs2), vals, with_subs=True)
    assert_true(allclose(nvals, array([1, 5])))
    assert_true(allclose(nsubs[0], array([0, 1])))
    assert_true(allclose(nsubs[1], array([0, 1])))

    subs1 = array([0, 0, 1])
    subs2 = array([0, 0, 1])
    vals = array([1, 2, 3])
    nvals, nsubs = accum((subs1, subs2), vals, with_subs=True)
    assert_true(allclose(nvals, array([3, 3])))
    assert_true(allclose(nsubs[0], array([0, 1])))
    assert_true(allclose(nsubs[1], array([0, 1])))
