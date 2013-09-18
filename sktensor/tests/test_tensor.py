import logging
from numpy.random import randn
from sktensor.tensor import *

from nose.tools import assert_equal
from nose.tools import assert_true


def test_new():
    sz = (10, 23, 5)
    A = randn(*sz)
    T = tensor(A)
    assert_equal(A.ndim, T.ndim)
    assert_equal(A.shape, T.shape)


def test_ttm():
    sz = (10, 35, 3, 12)
    rank = (3, 5, 2, 2)
    U = [randn(sz[i], rank[i]) for i in range(len(sz))]
    core = tensor(randn(*rank))
    T = core.ttm(U)
    T2 = ttm(core, U)
    assert_equal(sz, T.shape)
    assert_equal(sz, T2.shape)
    assert_true((T == T2).all())
