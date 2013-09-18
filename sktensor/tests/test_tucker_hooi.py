import logging
from numpy import allclose
from numpy.random import randn
from scipy.sparse import rand as sprand
from sktensor import tucker_hooi
from sktensor.core import ttm
from sktensor.sptensor import fold, sptensor
from sktensor.tensor import fold as tt_fold
from sktensor.rotation import orthomax

from nose.tools import assert_true

logging.basicConfig(level=logging.INFO)


def normalize(X):
   return X / X.sum(axis=0)


def est_factorization():
    I, J, K, rank = 10, 20, 75, 5
    A = normalize(randn(I, rank))
    B = normalize(randn(J, rank))
    C = normalize(randn(K, rank))

    core_real = randn(rank, rank, rank)
    T = ttm(core_real, [A, B, C])
    core, U = tucker_hooi.tucker_hooi(T, rank)

    assert_true(allclose(T, ttm(core, U)))
    assert_true(allclose(A, normalize(U[0])))
    assert_true(allclose(B, normalize(U[1])))
    assert_true(allclose(B, normalize(U[2])))
    assert_true(allclose(core_real, core))


def test_factorization_sparse():
    I, J, K, rank = 10, 20, 75, 5
    Tmat = sprand(I, J * K, 0.1).tocoo()
    T = fold((Tmat.row, Tmat.col), Tmat.data, 0, (I, J, K))
    core, U = tucker_hooi.tucker_hooi(T, rank, maxIter=20)

    Tmat = Tmat.toarray()
    T = tt_fold(Tmat, 0, (I, J, K))
    core2, U2 = tucker_hooi.tucker_hooi(T, rank, maxIter=20)

    assert_true(allclose(core2, core))
    for i in range(len(U)):
        assert_true(allclose(U2[i], U[i]))
