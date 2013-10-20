"""
sktensor.dtensor - base class for dense tensors
Copyright (C) 2013 Maximilian Nickel <maximilian.nickel@iit.it>

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
from numpy import array, prod, argsort
from sktensor.core import tensor_mixin


class dtensor(tensor_mixin, np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __eq__(self, other):
        return np.equal(self, other)

    def _ttm_compute(self, V, mode, transp):
        sz = array(self.shape)
        r1 = range(0, mode)
        r2 = range(mode + 1, self.ndim)
        order = [mode] + r1 + r2
        newT = np.transpose(self, axes=order)
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

    def __ttv_compute(T, v, dims, vidx, remdims):
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

    def unfold(self, mode):
        sz = array(self.shape)
        N = len(sz)
        #order = ([n], range(n) + range(n + 1, N))
        order = ([mode], range(N - 1, mode, -1) + range(mode - 1, -1, -1))
        newsz = (sz[order[0]], prod(sz[order[1]]))
        arr = np.transpose(self, axes=order[0] + order[1]).reshape(newsz)
        return unfolded_dtensor(arr, mode, self.shape)


class unfolded_dtensor(np.ndarray):

    def __new__(cls, input_array, mode, ten_shape):
        obj = np.asarray(input_array).view(cls)
        obj.ten_shape = ten_shape
        obj.mode = mode
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.ten_shape = getattr(obj, 'ten_shape', None)
        self.mode = getattr(obj, 'mode', None)

    def fold(self):
        shape = array(self.ten_shape)
        N = len(shape)
        #order = ([n], range(n) + range(n + 1, N))
        order = (
            [self.mode],
            range(N - 1, self.mode, -1) + range(self.mode - 1, -1, -1)
        )
        #print tuple(shape[order[0]],) + tuple(shape[order[1]])
        arr = self.reshape(tuple(shape[order[0]],) + tuple(shape[order[1]]))
        arr = np.transpose(arr, argsort(order[0] + order[1]))
        return dtensor(arr)
