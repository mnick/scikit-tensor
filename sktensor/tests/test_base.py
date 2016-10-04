from numpy import array
from numpy.random import randn
from sktensor.core import *
from sktensor import dtensor, sptensor, ktensor

from .fixtures import ttm_fixture, sptensor_fixture

ttm_fixture(__name__)
sptensor_fixture(__name__)


def test_check_multiplication_dims():
    ndims = 3
    M = 2
    assert ([1, 2] == check_multiplication_dims(0, ndims, M, without=True)).all()
    assert ([0, 2] == check_multiplication_dims(1, ndims, M, without=True)).all()
    assert ([0, 1] == check_multiplication_dims(2, ndims, M, without=True)).all()


def test_khatrirao():
    A = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    B = array([
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]
    ])
    C = array([
        [1, 8, 21],
        [2, 10, 24],
        [3, 12, 27],
        [4, 20, 42],
        [8, 25, 48],
        [12, 30, 54],
        [7, 32, 63],
        [14, 40, 72],
        [21, 48, 81]
    ])

    D = khatrirao((A, B))
    assert C.shape == D.shape
    assert (C == D).all()

def test_dense_fold():
    X = dtensor(T)
    I, J, K = T.shape
    X1 = X[:, :, 0]
    X2 = X[:, :, 1]

    U = X.unfold(0)
    assert (3, 8) == U.shape
    for j in range(J):
        assert (U[:, j] == X1[:, j]).all()
        assert (U[:, j + J] == X2[:, j]).all()

    U = X.unfold(1)
    assert (4, 6) == U.shape
    for i in range(I):
        assert (U[:, i] == X1[i, :]).all()
        assert (U[:, i + I] == X2[i, :]).all()

    U = X.unfold(2)
    assert (2, 12) == U.shape
    for k in range(U.shape[1]):
        assert (U[:, k] == array([X1.flatten('F')[k], X2.flatten('F')[k]])).all()


def test_dtensor_fold_unfold():
    sz = (10, 35, 3, 12)
    X = dtensor(randn(*sz))
    for i in range(4):
        U = X.unfold(i).fold()
        assert (X == U).all()


def test_dtensor_ttm():
    X = dtensor(T)
    Y2 = X.ttm(U, 0)
    assert (2, 4, 2) == Y2.shape
    assert (Y == Y2).all()


def test_spttv():
    #subs = (
    #    array([0, 1, 0, 5, 7, 8]),
    #    array([2, 0, 4, 5, 3, 9]),
    #    array([0, 1, 2, 2, 1, 0])
    #)
    #vals = array([1, 1, 1, 1, 1, 1])
    S = sptensor(subs, vals, shape=shape)
    K = ktensor([randn(shape[0], 2), randn(shape[1], 2), randn(shape[2], 2)])
    K.innerprod(S)
