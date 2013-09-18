import numpy as np
from numpy import vstack, ones, zeros, array, setdiff1d, allclose
from numpy.random import randint
from sktensor.core import unfold, ttm
from sktensor.sptensor import sptensor, fold, sttm_me_compute
from nose.tools import assert_equal, assert_true, raises
from fixtures import ttm_fixture

ttm_fixture(__name__)

def mysetup():
    shape = (25, 11, 18, 7, 2)
    sz = 100
    subs = [None for _ in xrange(len(shape))]
    for i in range(len(shape)):
        subs[i] = randint(0, shape[i], sz)
    #subs = vstack(subs).T
    vals = ones(sz)
    return subs, vals, shape


def test_init():
    subs, vals, shape = mysetup()
    T = sptensor(subs, vals, shape)
    assert_equal(len(shape), T.ndim)
    assert_true((array(shape) == T.shape).all())

    T = sptensor(subs, vals)
    tshape = array(subs).max(axis=1)
    assert_equal(len(subs), len(T.shape))
    assert_true((tshape == array(T.shape)).all())


@raises(ValueError)
def test_non2Dsubs():
    sptensor(randint(0, 10, 18).reshape(3, 3, 2), ones(10))


@raises(ValueError)
def test_nonEqualLength():
    subs, vals, shape = mysetup()
    sptensor(subs, ones(len(subs) + 1))


def test_unfold():
    subs, vals, shape = mysetup()
    Td = zeros(shape, dtype=np.float32)
    Td[subs] = 1

    for i in range(len(shape)):
        rdims = [i]
        cdims = setdiff1d(range(len(shape)), rdims)[::-1]
        Md = unfold(Td, i)

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
    subs, vals, shape = mysetup()
    T = sptensor(subs, vals, shape)
    for i in range(len(shape)):
        Ms = T.unfold([i])
        T = fold((Ms.row, Ms.col), Ms.data, [i], shape)
        assert_equal(shape, tuple(T.shape))
        assert_equal(len(shape), len(T.subs))
        assert_equal(len(subs), len(T.subs))
        for j in xrange(len(subs)):
            assert_true((subs[j] == T.subs[j]).all())


def test_ttm():
    T = globals()['T']
    T = sptensor(T.nonzero(), T.flatten(), T.shape)
    Y2 = ttm(T, U, 0)
    assert_equal((2, 4, 2), Y2.shape)
    assert_true((Y == Y2).all())


def test_sttm_me():
    T = globals()['T']
    print T.shape
    T = sptensor(T.nonzero(), T.flatten(), T.shape)
    sttm_me_compute(T, U, [1], [0], False)
