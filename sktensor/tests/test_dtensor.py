from numpy import array
from numpy.random import randn
from sktensor.dtensor import dtensor

from .fixtures import ttm_fixture

ttm_fixture(__name__)


def test_new():
    sz = (10, 23, 5)
    A = randn(*sz)
    T = dtensor(A)
    assert A.ndim == T.ndim
    assert A.shape == T.shape
    assert (A == T).all()
    assert (T == A).all()


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
