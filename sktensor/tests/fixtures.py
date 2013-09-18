from numpy import array, zeros
import sys


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
