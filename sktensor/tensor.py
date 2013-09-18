"""
sktensor.tensor - base class for dense tensors
Copyright (C) 2013 Maximilian Nickel <nickel@dbs.ifi.lmu.de>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from numpy import array, argsort, prod
#from sktensor.core import ttm, ttv


class tensor(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def ttm(self, V, mode=None, transp=False, without=False):
        """
        Tensor times matrix product
        """
        return ttm(self, V, mode, transp, without)

    def ttv(self, v, dims=[]):
        return ttv(self, v, dims)


def unfold(X, n):
    sz = array(X.shape)
    N = len(sz)
    #order = ([n], range(n) + range(n + 1, N))
    order = ([n], range(N - 1, n, -1) + range(n - 1, -1, -1))
    newsz = (sz[order[0]], prod(sz[order[1]]))
    return np.transpose(X, axes=order[0] + order[1]).reshape(newsz)


def fold(Xn, n, shape):
    shape = array(shape)
    N = len(shape)
    #order = ([n], range(n) + range(n + 1, N))
    order = ([n], range(N - 1, n, -1) + range(n - 1, -1, -1))
    #print tuple(shape[order[0]],) + tuple(shape[order[1]])
    X = Xn.reshape(tuple(shape[order[0]],) + tuple(shape[order[1]]))
    return np.transpose(X, argsort(order[0] + order[1]))


def __ttm_compute(T, V, mode, transp):
    sz = array(T.shape)
    N = T.ndim
    r1 = range(0, mode)
    r2 = range(mode + 1, N)
    order = [mode] + r1 + r2
    newT = np.transpose(T, axes=order)
    newT = newT.reshape(sz[mode], prod(sz[r1 + range(mode + 1, len(sz))]))
    if transp:
        newT = V.T.dot(newT)
        p = V.shape[1]
    else:
        newT = V.dot(newT)
        p = V.shape[0]
    newsz = [p] + list(sz[:mode]) + list(sz[mode + 1:])
    newT = newT.reshape(newsz)
    # transpose + argsort(order) equals ipermute
    return np.transpose(newT, argsort(order))


def ttv(T, v, dims, vidx, remdims):
    """
    Tensor times vector product

    Parameter
    ---------
    """
    if not isinstance(v, tuple):
        raise ValueError('v must be a tuple of vectors')
    ndim = T.ndim
    order = list(remdims) + list(dims)
    if ndim > 1:
        T = np.transpose(T, order)
    sz = array(T.shape)[order]
    for i in np.arange(len(dims), 0, -1):
        T = T.reshape((sz[:ndim - 1].prod(), sz[ndim - 1]))
        T = T.dot(v[vidx[i - 1]])
        ndim -= 1
    if ndim > 0:
        T = T.reshape(sz)
    return T
