"""
sktensor.sptensor - base module for sparse tensors
Copyright (C) 2013 Maximilian Nickel <mnick@mit.edu>

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
from numpy import zeros, ones, array, arange, copy, ravel_multi_index, unravel_index
from numpy import setdiff1d, hstack, hsplit, vsplit, sort, prod, lexsort
from scipy.sparse import coo_matrix
from scipy.sparse import issparse as issparse_mat
from sktensor.core import tensor_mixin
from sktensor.utils import accum
from sktensor.dtensor import unfolded_dtensor


__all__ = [
    'concatenate',
    'fromarray',
    'issparse',
    'sptensor',
    'unfolded_sptensor',
]


class sptensor(tensor_mixin):
    """
    Sparse tensor class. Stores data in COOrdinate format.

    Sparse tensors can be instantiated via

    Parameters
    ----------
    subs : n-tuple of array-likes
        Subscripts of the nonzero entries in the tensor.
        Length of tuple n must be equal to dimension of tensor.
    vals : array-like
        Values of the nonzero entries in the tensor.
    shape : n-tuple, optional
        Shape of the sparse tensor.
        Length of tuple n must be equal to dimension of tensor.
    dtype : dtype, optional
        Type of the entries in the tensor
    accumfun : function pointer
        Function to be accumulate duplicate entries

    Examples
    --------
    >>> S = sptensor(([0,1,2], [3,2,0], [2,2,2]), [1,1,1], shape=(10, 20, 5), dtype=np.float)
    >>> S.shape
    (10, 20, 5)
    >>> S.dtype
    <type 'float'>

    """

    def __init__(self, subs, vals, shape=None, dtype=None, accumfun=None):
        if not isinstance(subs, tuple):
            raise ValueError('Subscripts must be a tuple of array-likes')
        if len(subs[0]) != len(vals):
            raise ValueError('Subscripts and values must be of equal length')
        if dtype is None:
            dtype = array(vals).dtype
        for i in range(len(subs)):
            if array(subs[i]).dtype.kind != 'i':
                raise ValueError('Subscripts must be integers')

        vals = array(vals)
        if accumfun is not None:
            vals, subs = accum(
                subs, vals,
                sorted=False, with_subs=True, fun=accumfun
            )
        self.subs = subs
        self.vals = vals
        self.dtype = dtype

        if shape is None:
            self.shape = tuple(array(subs).max(axis=1).flatten())
        else:
            self.shape = tuple(int(d) for d in shape)
        self.ndim = len(subs)

    def __eq__(self, other):
        self._sort()
        other._sort()
        return (self.vals == other.vals).all() and (array(self.subs) == array(other.subs)).all()

    def _sort(self):
        subs = array(self.subs)
        sidx = lexsort(subs)
        self.subs = tuple(z.flatten()[sidx] for z in vsplit(subs, len(self.shape)))
        self.vals = self.vals[sidx]

    def _ttm_compute(self, V, mode, transp):
        Z = self.unfold(mode, transp=True).tocsr()
        if transp:
            V = V.T
        Z = Z.dot(V.T)
        shape = copy(self.shape)
        shape[mode] = V.shape[0]
        if issparse_mat(Z):
            newT = unfolded_sptensor((Z.data, (Z.row, Z.col)), [mode], None, shape=shape).fold()
        else:
            newT = unfolded_dtensor(Z.T, mode, shape).fold()

        return newT

    def _ttv_compute(self, v, dims, vidx, remdims):
        nvals = self.vals
        nsubs = self.subs
        for i in xrange(len(dims)):
            idx = nsubs[dims[i]]
            w = v[vidx[i]]
            nvals = nvals * w[idx]

        # Case 1: all dimensions used -> return sum
        if len(remdims) == 0:
            return nvals.sum()

        nsubs = tuple(self.subs[i] for i in remdims)
        nshp = tuple(self.shape[i] for i in remdims)

        # Case 2: result is a vector
        if len(remdims) == 1:
            c = accum(nsubs, nvals, shape=nshp)
            if len(np.nonzero(c)[0]) <= 0.5 * prod(nshp):
                return sptensor(arange(nshp), c)
            else:
                return c

        # Case 3: result is an array
        return sptensor(nsubs, nvals, shape=nshp, accumfun=np.sum)

    def sttm_me_compute(self, V, edims, sdims, transp):
        """
        Assume Y = T x_i V_i for i = 1...n can fit into memory
        """
        shapeY = self.shape.copy()

        # Determine size of Y
        for n in np.union1d(edims, sdims):
            shapeY[n] = V[n].shape[1] if transp else V[n].shape[0]

        # Allocate Y (final result) and v (vectors for elementwise computations)
        Y = zeros(shapeY)
        shapeY = array(shapeY)
        v = [None for _ in xrange(len(edims))]

        for i in xrange(np.prod(shapeY[edims])):
            rsubs = unravel_index(shapeY[edims], i)

    def unfold(self, rdims, cdims=None, transp=False):
        if isinstance(rdims, type(1)):
            rdims = [rdims]
        if transp:
            cdims = rdims
            rdims = setdiff1d(range(self.ndim), cdims)[::-1]
        elif cdims is None:
            cdims = setdiff1d(range(self.ndim), rdims)[::-1]
        if not (arange(self.ndim) == sort(hstack((rdims, cdims)))).all():
            raise ValueError(
                'Incorrect specification of dimensions (rdims: %s, cdims: %s)'
                % (str(rdims), str(cdims))
            )
        M = prod(self.shape[rdims])
        N = prod(self.shape[cdims])
        ridx = _build_idx(self.subs, self.vals, rdims, self.shape)
        cidx = _build_idx(self.subs, self.vals, cdims, self.shape)
        return unfolded_sptensor((self.vals, (ridx, cidx)), (M, N), rdims, cdims, self.shape)

    def uttkrp(self, U, mode):
        R = U[1].shape[1] if mode == 0 else U[0].shape[1]
        dims = range(0, mode) + range(mode + 1, self.ndim)
        V = zeros((self.shape[mode], R))
        for r in xrange(R):
            Z = tuple(U[n][:, r] for n in dims)
            V[:, r] = self.ttv(Z, mode, without=True)
        return V


    def transpose(self, axes=None):
        if axes is None:
            raise NotImplementedError(
                'Sparse tensor transposition without axes argument is not supported'
            )
        nsubs = tuple([self.subs[idx] for idx in axes])
        nshape = [self.shape[idx] for idx in axes]
        return sptensor(nsubs, self.vals, nshape)


    def norm(self):
        """
        Frobenius norm for tensors

        References
        ----------
        [Kolda and Bader, 2009; p.457]
        """
        return np.linalg.norm(self.vals)


    def toarray(self):
        A = zeros(self.shape)
        A.put(ravel_multi_index(self.subs, tuple(self.shape)), self.vals)
        return A


class unfolded_sptensor(coo_matrix):

    def __init__(self, tpl, shape, rdims, cdims, ten_shape, dtype=None, copy=False):
        self.ten_shape = array(ten_shape)
        if isinstance(rdims, int):
            rdims = [rdims]
        if cdims is None:
            cdims = setdiff1d(range(len(self.ten_shape)), rdims)[::-1]
        self.rdims = rdims
        self.cdims = cdims
        super(unfolded_sptensor, self).__init__(tpl, shape=shape, dtype=dtype, copy=copy)

    def fold(self):
        nsubs = zeros((len(self.data), len(self.ten_shape)), dtype=np.int)
        if len(self.rdims) > 0:
            nidx = unravel_index(self.row, self.ten_shape[self.rdims])
            for i in xrange(len(self.rdims)):
                nsubs[:, self.rdims[i]] = nidx[i]
        if len(self.cdims) > 0:
            nidx = unravel_index(self.col, self.ten_shape[self.cdims])
            for i in xrange(len(self.cdims)):
                nsubs[:, self.cdims[i]] = nidx[i]
        nsubs = [z.flatten() for z in hsplit(nsubs, len(self.ten_shape))]
        return sptensor(tuple(nsubs), self.data, self.ten_shape)


def fromarray(A):
    """Create a sptensor from a dense numpy array"""
    subs = np.nonzero(A)
    vals = A[subs]
    return sptensor(subs, vals, shape=A.shape, dtype=A.dtype)


def concatenate(tpl, axis=None):
    """
    Concatenate sparse tensors

    Parameter
    ---------
    tpl :  tuple of sparse tensors
        Tensors to be concatenated.
    axis :  int, optional
        Axis along which concatenation should take place
    """
    if axis is None:
        raise NotImplementedError(
            'Sparse tensor concatenation without axis argument is not supported'
        )
    T = tpl[0]
    for i in range(1, len(tpl)):
        T = _single_concatenate(T, tpl[i], axis=axis)
    return T


def _single_concatenate(ten, other, axis):
    tshape = ten.shape
    oshape = other.shape
    if len(tshape) != len(oshape):
        raise ValueError("len(tshape) != len(oshape")
    oaxes = setdiff1d(range(len(tshape)), [axis])
    for i in oaxes:
        if tshape[i] != oshape[i]:
            raise ValueError("Dimensions must match")
    nsubs = [None for _ in xrange(len(tshape))]
    for i in oaxes:
        nsubs[i] = np.concatenate((ten.subs[i], other.subs[i]))
    nsubs[axis] = np.concatenate((
        ten.subs[axis], other.subs[axis] + tshape[axis]
    ))
    nvals = np.concatenate((ten.vals, other.vals))
    nshape = np.copy(tshape)
    nshape[axis] = tshape[axis] + oshape[axis]
    return sptensor(nsubs, nvals, nshape)


def _build_idx(subs, vals, dims, tshape):
    shape = array(tshape[dims], ndmin=1)
    dims = array(dims, ndmin=1)
    if len(shape) == 0:
        idx = ones(len(vals), dtype=vals.dtype)
    elif len(subs) == 0:
        idx = array(tuple())
    else:
        idx = ravel_multi_index(tuple(subs[i] for i in dims), shape)
    return idx


#def ttv(T, v, dims, vidx, remdims):
#    if not isinstance(v, tuple):
#        raise ValueError('v must be a tuple of vectors')
#    nvals = T.vals
#    for n in range(len(dims)):
#        idx = T.subs[dims[n]]
#        w = v[vidx[n]]
#        W = w[idx]
#        nvals = nvals * W
#
#    # all dimensions used, return sum
#    if len(remdims) == 0:
#        return nvals.sum()
#
#    # otherwise accumulate
#    nsubs = [T.subs[r] for r in remdims]
#    nsz = [T.shape[r] for r in remdims]
#
#    # result is a vector
#    if len(remdims) == 1:
#        c = accum(nsubs, nvals, nsz)
#        if len(np.nonzero(c)[0]) <= 0.5 * nsz:
#            return sptensor(arange(nsz), c)
#        else:
#            return nvals
#
#    # result is an array
#    return sptensor(nsubs, nvals, shape=nsz, accumfun=np.sum)
