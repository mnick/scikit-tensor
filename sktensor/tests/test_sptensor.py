import pytest
import numpy as np
from numpy import ones, zeros, array, setdiff1d, allclose
from numpy.random import randint
from sktensor.dtensor import dtensor
from sktensor.sptensor import sptensor, fromarray
from .ttm_fixture import T, U, Y
from .sptensor_rand_fixture import subs, vals, shape, sptensor_seed, sz


def setup_diagonal():
    """
    Setup data for a 20x20x20 diagonal tensor
    """
    n = 20
    shape = (n, n, n)
    subs = [np.arange(0, shape[i]) for i in range(len(shape))]
    vals = ones(n)
    return tuple(subs), vals, shape


def test_init(subs, vals, shape):
    """
    Creation of new sptensor objects
    """
    T = sptensor(subs, vals, shape)
    assert len(shape) == T.ndim
    assert (array(shape) == T.shape).all()

    T = sptensor(subs, vals)
    tshape = array(subs).max(axis=1) + 1
    assert len(subs) == len(T.shape)
    assert (tshape == array(T.shape)).all()


def test_init_diagonal():
    subs, vals, shape = setup_diagonal()
    T = sptensor(subs, vals, shape)
    assert len(shape) == T.ndim
    assert (array(shape) == T.shape).all()

    T = sptensor(subs, vals)
    assert len(subs) == len(T.shape)
    assert (shape == array(T.shape)).all()


def test_non2Dsubs():
    with pytest.raises(ValueError):
        sptensor(randint(0, 10, 18).reshape(3, 3, 2), ones(10))


def test_nonEqualLength(subs):
    with pytest.raises(ValueError):
        sptensor(subs, ones(len(subs) + 1))


def test_unfold(T, subs, vals, shape):
    Td = dtensor(zeros(shape, dtype=np.float32))
    Td[subs] = vals

    for i in range(len(shape)):
        rdims = [i]
        cdims = setdiff1d(range(len(shape)), rdims)[::-1]
        Md = Td.unfold(i)

        T = sptensor(subs, vals, shape, accumfun=lambda l: l[-1])

        Ms = T.unfold(rdims, cdims)
        assert Md.shape == Ms.shape
        assert (allclose(Md, Ms.toarray()))

        Ms = T.unfold(rdims)
        assert Md.shape == Ms.shape
        assert (allclose(Md, Ms.toarray()))

        Md = Md.T
        Ms = T.unfold(rdims, cdims, transp=True)
        assert Md.shape == Ms.shape
        assert (allclose(Md, Ms.toarray()))


def test_fold(subs, vals, shape):
    T = sptensor(subs, vals, shape)
    for i in range(len(shape)):
        X = T.unfold([i]).fold()
        assert shape == tuple(T.shape)
        assert len(shape) == len(T.subs)
        assert len(subs) == len(T.subs)
        assert X == T
        for j in range(len(subs)):
            subs[j].sort()
            T.subs[j].sort()
            assert (subs[j] == T.subs[j]).all()


def test_ttm(T, Y, U):
    S = sptensor(T.nonzero(), T.flatten(), T.shape)
    Y2 = S.ttm(U, 0)
    assert (2, 4, 2) == Y2.shape
    assert (Y == Y2).all()


def test_ttv_sparse_result():
    # Test case by Andre Panisson to check return type of sptensor.ttv
    subs = (
        array([0, 1, 0, 5, 7, 8]),
        array([2, 0, 4, 5, 3, 9]),
        array([0, 1, 2, 2, 1, 0])
    )
    vals = array([1, 1, 1, 1, 1, 1])
    S = sptensor(subs, vals, shape=[10, 10, 3])

    sttv = S.ttv((zeros(10), zeros(10)), modes=[0, 1])
    assert type(sttv) == sptensor
    # sparse tensor should return only nonzero vals
    assert (allclose(np.array([]), sttv.vals))
    assert (allclose(np.array([]), sttv.subs))
    assert sttv.shape == (3,)


def test_ttv(T):
    result = array([
        [70, 190],
        [80, 200],
        [90, 210]
    ])

    X = fromarray(T)
    v = array([1, 2, 3, 4])
    Xv = X.ttv(v, 1)

    assert (3, 2) == Xv.shape
    assert (Xv == result).all()


def test_sttm_me(T, U):
    S = sptensor(T.nonzero(), T.flatten(), T.shape)
    S._ttm_me_compute(U, [1], [0], False)


def test_sp_uttkrp(subs, vals, shape):
    # Test case by Andre Panisson, sparse ttv
    # see issue #3
    S = sptensor(subs, vals, shape)
    U = []
    for shp in shape:
        U.append(np.zeros((shp, 5)))
    SU = S.uttkrp(U, mode=0)
    assert SU.shape == (25, 5)


def test_getitem():
    subs = (
        array([0, 1, 0, 5, 7, 8]),
        array([2, 0, 4, 5, 3, 9]),
        array([0, 1, 2, 2, 1, 0])
    )
    vals = array([1, 2, 3, 4, 5, 6])
    S = sptensor(subs, vals, shape=[10, 10, 3])
    assert 0 == S[1, 1, 1]
    assert 0 == S[1, 2, 3]
    assert 1 == S[0, 2, 0]
    assert 2 == S[1, 0, 1]
    assert 3 == S[0, 4, 2]
    assert 4 == S[5, 5, 2]
    assert 5 == S[7, 3, 1]
    assert 6 == S[8, 9, 0]


def test_add():
    subs = (
        array([0, 1, 0]),
        array([2, 0, 2]),
        array([0, 1, 2])
    )
    vals = array([1, 2, 3])
    S = sptensor(subs, vals, shape=[3, 3, 3])
    D = np.arange(27).reshape(3, 3, 3)
    T = S - D
    for i in range(3):
        for j in range(3):
            for k in range(3):
                assert S[i, j, k] - D[i, j, k] == T[i, j, k]
