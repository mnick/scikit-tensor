# sktensor.dtensor - base class for dense tensors
# Copyright (C) 2013 Maximilian Nickel <mnick@mit.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from numpy import array, prod, argsort
from sktensor.core import tensor_mixin, khatrirao


class dtensor(tensor_mixin, np.ndarray):
    """
    Class to store **dense** tensors

    Parameters
    ----------
    input_array : np.ndarray
        Multidimenional numpy array which holds the entries of the tensor

    Examples
    --------
    Create dense tensor from numpy array

    >>> T = np.zeros((3, 4, 2))
    >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
    >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
    >>> T = dtensor(T)
    """

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
        newT = np.transpose(newT, argsort(order))
        return dtensor(newT)

    def _ttv_compute(self, v, dims, vidx, remdims):
        """
        Tensor times vector product

        Parameter
        ---------
        """
        if not isinstance(v, tuple):
            raise ValueError('v must be a tuple of vectors')
        ndim = self.ndim
        order = list(remdims) + list(dims)
        if ndim > 1:
            T = np.transpose(self, order)
        sz = array(self.shape)[order]
        for i in np.arange(len(dims), 0, -1):
            T = T.reshape((sz[:ndim - 1].prod(), sz[ndim - 1]))
            T = T.dot(v[vidx[i - 1]])
            ndim -= 1
        if ndim > 0:
            T = T.reshape(sz)
        return T

    def unfold(self, mode):
        """
        Unfolds a dense tensor in mode n.

        Parameters
        ----------
        mode : int
            Mode in which tensor is unfolded

        Returns
        -------
        unfolded_dtensor : unfolded_dtensor object
            Tensor unfolded along mode

        Examples
        --------
        Create dense tensor from numpy array

        >>> T = np.zeros((3, 4, 2))
        >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
        >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
        >>> T = dtensor(T)

        Unfolding of dense tensors

        >>> T.unfold(0)
        array([[  1.,   4.,   7.,  10.,  13.,  16.,  19.,  22.],
               [  2.,   5.,   8.,  11.,  14.,  17.,  20.,  23.],
               [  3.,   6.,   9.,  12.,  15.,  18.,  21.,  24.]])
        >>> T.unfold(1)
        array([[  1.,   2.,   3.,  13.,  14.,  15.],
               [  4.,   5.,   6.,  16.,  17.,  18.],
               [  7.,   8.,   9.,  19.,  20.,  21.],
               [ 10.,  11.,  12.,  22.,  23.,  24.]])
        >>> T.unfold(2)
        array([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,
                 12.],
               [ 13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,  22.,  23.,
                 24.]])
        """

        sz = array(self.shape)
        N = len(sz)
        #order = ([n], range(n) + range(n + 1, N))
        order = ([mode], range(N - 1, mode, -1) + range(mode - 1, -1, -1))
        newsz = (sz[order[0]], prod(sz[order[1]]))
        arr = transpose(self, axes=(order[0] + order[1]))
        arr = arr.reshape(newsz)
        return unfolded_dtensor(arr, mode, self.shape)

    def norm(self):
        """
        Frobenius norm for tensors

        References
        ---
        [Kolda and Bader, 2009; p.457]
        """
        return np.linalg.norm(self)

    def uttkrp(self, U, n):
        """
        Unfolded tensor times Khatri-Rao product for dense tensors

        Parameters
        ----------
        X : tensor_mixin
        U : list of array-like
        n : int

        See also
        --------
        For efficient computations unfolded tensor times Khatri-Rao products
        for specialized tensors see also
        - :func::`sptensor.uttkrp`
        - ktensor.uttkrp
        - ttensor.uttkrp

        References
        ----------
        [1] B.W. Bader, T.G. Kolda
            Efficient Matlab Computations With Sparse and Factored Tensors
            SIAM J. Sci. Comput, Vol 30, No. 1, pp. 205--231, 2007
        """
        order = range(n) + range(n + 1, self.ndim)
        Z = khatrirao(tuple(U[i] for i in order), reverse=True)
        return self.unfold(n).dot(Z)


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

def transpose(dten, axes=None):
    """
    Compute transpose of dense tensor

    Parameters
    ----------
    dten : dtensor
        Dense tensor to be transposed.
    axes : array_like of ints, optional
        Permute the axes according to the values given.

    Returns
    -------
    d : dtensor
        dtensor with axes permuted.
    """
    return dtensor(np.transpose(array(dten), axes=axes))
