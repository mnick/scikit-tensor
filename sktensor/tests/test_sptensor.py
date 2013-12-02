import numpy as np
from numpy import ones, zeros, array, setdiff1d, allclose
from numpy.random import randint
from sktensor.dtensor import dtensor
from sktensor.sptensor import sptensor
from nose.tools import assert_equal, assert_true, raises
from .fixtures import ttm_fixture, sptensor_rand_fixture

ttm_fixture(__name__)
sptensor_rand_fixture(__name__)


def setup_diagonal():
    """
    Setup data for a 20x20x20 diagonal tensor
    """
    n = 20
    shape = (n, n, n)
    subs = [np.arange(0, shape[i]) for i in range(len(shape))]
    vals = ones(n)
    return tuple(subs), vals, shape


def test_init():
    """
    Creation of new sptensor objects
    """
    T = sptensor(subs, vals, shape)
    assert_equal(len(shape), T.ndim)
    assert_true((array(shape) == T.shape).all())

    T = sptensor(subs, vals)
    tshape = array(subs).max(axis=1) + 1
    assert_equal(len(subs), len(T.shape))
    assert_true((tshape == array(T.shape)).all())


def test_init_diagonal():
    subs, vals, shape = setup_diagonal()
    T = sptensor(subs, vals, shape)
    assert_equal(len(shape), T.ndim)
    assert_true((array(shape) == T.shape).all())

    T = sptensor(subs, vals)
    assert_equal(len(subs), len(T.shape))
    assert_true((shape == array(T.shape)).all())


@raises(ValueError)
def test_non2Dsubs():
    sptensor(randint(0, 10, 18).reshape(3, 3, 2), ones(10))


@raises(ValueError)
def test_nonEqualLength():
    sptensor(subs, ones(len(subs) + 1))


def test_unfold():
    Td = dtensor(zeros(shape, dtype=np.float32))
    Td[subs] = vals

    for i in range(len(shape)):
        rdims = [i]
        cdims = setdiff1d(range(len(shape)), rdims)[::-1]
        Md = Td.unfold(i)

        T = sptensor(subs, vals, shape)

        Ms = T.unfold(rdims, cdims)
        assert_equal(Md.shape, Ms.shape)
        assert_true((allclose(Md, Ms.toarray())))

        Ms = T.unfold(rdims)
        assert_equal(Md.shape, Ms.shape)
        assert_true((allclose(Md, Ms.toarray())))

        Md = Md.T
        Ms = T.unfold(rdims, cdims, transp=True)
        assert_equal(Md.shape, Ms.shape)
        assert_true((allclose(Md, Ms.toarray())))


def test_fold():
    T = sptensor(subs, vals, shape)
    for i in range(len(shape)):
        X = T.unfold([i]).fold()
        assert_equal(shape, tuple(T.shape))
        assert_equal(len(shape), len(T.subs))
        assert_equal(len(subs), len(T.subs))
        assert_equal(X, T)
        for j in range(len(subs)):
            subs[j].sort()
            T.subs[j].sort()
            assert_true((subs[j] == T.subs[j]).all())


def test_ttm():
    S = sptensor(T.nonzero(), T.flatten(), T.shape)
    Y2 = S.ttm(U, 0)
    assert_equal((2, 4, 2), Y2.shape)
    assert_true((Y == Y2).all())


def test_ttv():
    # Test case by Andre Panisson to check return type of sptensor.ttv
    subs = (
        array([0, 1, 0, 5, 7, 8]),
        array([2, 0, 4, 5, 3, 9]),
        array([0, 1, 2, 2, 1, 0])
    )
    vals = array([1, 1, 1, 1, 1, 1])
    S = sptensor(subs, vals, shape=[10, 10, 3])

    sttv = S.ttv((zeros(10), zeros(10)), modes=[0, 1])
    assert_equal(type(sttv), sptensor)
    assert_true((allclose(zeros(3), sttv.vals)))
    assert_true((allclose(np.arange(3), sttv.subs)))


def test_sttm_me():
    S = sptensor(T.nonzero(), T.flatten(), T.shape)
    S.ttm_me(U, [1], [0], False)


def test_sp_uttkrp():
    # Test case by Andre Panisson, sparse ttv
    # see issue #3
    S = sptensor(subs, vals, shape)
    U = []
    for shp in shape:
        U.append(np.zeros((shp, 5)))
    SU = S.uttkrp(U, mode=0)
    assert_equal(SU.shape, (25, 5))
