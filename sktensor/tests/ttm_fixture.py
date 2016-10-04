from numpy import array, zeros
import pytest


@pytest.fixture
def T():
    T = zeros((3, 4, 2))
    T[:, :, 0] = array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
    T[:, :, 1] = array([[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]])
    return T


@pytest.fixture
def Y():
    Y = zeros((2, 4, 2))
    Y[:, :, 0] = array([[22, 49, 76, 103], [28, 64, 100, 136]])
    Y[:, :, 1] = array([[130, 157, 184, 211], [172, 208, 244, 280]])
    return Y


@pytest.fixture
def U():
    return array([[1, 3, 5], [2, 4, 6]])
