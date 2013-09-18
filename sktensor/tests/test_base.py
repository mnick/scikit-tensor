from numpy import array, zeros
from numpy.random import randn
from sktensor.core import *
from sktensor import sptensor, ktensor

from nose import with_setup
from nose.tools import assert_true
from nose.tools import assert_equal
from fixtures import ttm_fixture

ttm_fixture(__name__)

def test_dimscheck():
    ndims = 3
    M = 2
    assert_true(([1, 2] == dimscheck(0, ndims, M, without=True)).all())
    assert_true(([0, 2] == dimscheck(1, ndims, M, without=True)).all())
    assert_true(([0, 1] == dimscheck(2, ndims, M, without=True)).all())


def test_fold():
    I, J, K = T.shape
    X1 = T[:, :, 0]
    X2 = T[:, :, 1]

    U = unfold(T, 0)
    assert_equal((3, 8), U.shape)
    for j in range(J):
        assert_true((U[:, K * j] == X1[:, j]).all())
        assert_true((U[:, K * j + 1] == X2[:, j]).all())

    U = unfold(T, 1)
    assert_equal((4, 6), U.shape)
    for i in range(I):
        assert_true((U[:, K * i] == X1[i, :]).all())
        assert_true((U[:, K * i + 1] == X2[i, :]).all())

    U = unfold(T, 2)
    assert_equal((2, 12), U.shape)
    assert_true((U[0] == X1.flatten()).all())
    assert_true((U[1] == X2.flatten()).all())


def test_fold_unfold():
    sz = (10, 35, 3, 12)
    T = randn(*sz)
    for i in range(4):
        U = fold(unfold(T, i), i, sz)
        assert_true((T == U).all())


def test_ttm():
    Y2 = ttm(T, U, 0)
    assert_equal((2, 4, 2), Y2.shape)
    assert_true((Y == Y2).all())


def test_spttv():
    subs = (
        array([0, 1, 0, 5, 7, 8]),
        array([2, 0, 4, 5, 3, 9]),
        array([0, 1, 2, 2, 1, 0])
    )
    vals = array([1, 1, 1, 1, 1, 1])
    S = sptensor(subs, vals, shape=[10, 10, 3])
    K = ktensor([randn(10, 2), randn(10, 2), randn(3, 2)])
    K.innerprod(S)
