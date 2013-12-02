import sys
from numpy import array, zeros
from numpy.random import randint


def ttm_fixture(mname):
    module = sys.modules[mname]

    T = zeros((3, 4, 2))
    T[:, :, 0] = array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
    T[:, :, 1] = array([[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]])
    module.T = T

    Y = zeros((2, 4, 2))
    Y[:, :, 0] = array([[22, 49, 76, 103], [28, 64, 100, 136]])
    Y[:, :, 1] = array([[130, 157, 184, 211], [172, 208, 244, 280]])
    module.Y = Y
    module.U = array([[1, 3, 5], [2, 4, 6]])


def sptensor_fixture(mname):
    module = sys.modules[mname]
    module.subs = (
        array([0, 1, 0, 5, 7, 8]),
        array([2, 0, 4, 5, 3, 9]),
        array([0, 1, 2, 2, 1, 0])
    )

    module.vals = array([1, 2, 3, 4, 5, 6.1])
    module.shape = (10, 12, 3)


def sptensor_rand_fixture(mname):
    shape = (25, 11, 18, 7, 2)
    sz = 100
    subs = tuple(randint(0, shape[i], sz) for i in range(len(shape)))

    module = sys.modules[mname]
    module.vals = randint(0, 100, sz)
    module.shape = shape
    module.subs = subs
