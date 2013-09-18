"""
sktensor.sptensor - base module for sparse tensors
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
from numpy import zeros, ones, array, arange, copy, ravel_multi_index
from numpy import setdiff1d, hstack, hsplit, sort, prod
from scipy.sparse import coo_matrix
from scipy.sparse import issparse as issparse_mat
from sktensor.utils import ravel_index, unravel_index, accum
import sktensor.tensor as tt


class sptensor(object):
    """
    Sparse tensor class. Stores data in COO format.

    >>> S = sptensor(([0,1,2], [3,2,0], [2,2,2]), [1,1,1])

    """

    def __init__(self, subs, vals, shape=None, dtype=None, accumfun=None):
        if not isinstance(subs, tuple):
            raise ValueError('subs must be a tuple of array-likes')
        if len(subs[0]) != len(vals):
            raise ValueError('subs and vals must be of equal length')
        if dtype is None:
            dtype = vals.dtype
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
            self.shape = array(subs).max(axis=1).flatten()
        else:
            self.shape = array(shape, dtype=np.int)
        self.ndim = len(subs)

    def unfold(self, rdims, cdims=None, transp=False):
        if isinstance(rdims, type(1)):
            rdims = [rdims]
        if transp:
            cdims = rdims
            rdims = setdiff1d(range(self.ndim), cdims)[::-1]
        elif cdims is None:
            cdims = setdiff1d(range(self.ndim), rdims)[::-1]
        if not (arange(self.ndim) == sort(hstack((rdims, cdims)))).all():
            raise ValueError('Incorrect specification of dimensions (rdims: %s, cdims: %s)' % (str(rdims), str(cdims)))
        return unfold(self.subs, self.vals, rdims, cdims, self.shape)

    def toarray(self):
        A = zeros(self.shape)
        A.put(ravel_multi_index(self.subs, tuple(self.shape)), self.vals)
        return A


def fromarray(A):
    """Create a sptensor from a dense numpy array"""
    subs = np.nonzero(A)
    vals = A[subs]
    return sptensor(subs, vals, shape=A.shape, dtype=A.dtype)


def transpose(T, axes=None):
    if axes is None:
        raise NotImplementedError("Sparse tensor transposition without axes argument is not supported")
    nsubs = tuple([T.subs[idx] for idx in axes])
    nshape = [T.shape[idx] for idx in axes]
    return sptensor(nsubs, T.vals, nshape)


def concatenate(tpl, axis):
    if axis == None:
        raise NotImplementedError("Sparse tensor concatenation without axis argument is not supported")
    T = tpl[0]
    for i in range(1, len(tpl)):
        T = __single_concatenate(T, tpl[i], axis=axis)
    return T


def __single_concatenate(ten, other, axis):
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
    nsubs[axis] = np.concatenate((ten.subs[axis], other.subs[axis] + tshape[axis]))
    nvals = np.concatenate((ten.vals, other.vals))
    nshape = np.copy(tshape)
    nshape[axis] = tshape[axis] + oshape[axis]
    return sptensor(nsubs, nvals, nshape)


def unfold(subs, vals, rdims, cdims, tshape):
    #subs = copy(subs)
    #vals = copy(vals)
    #tshape = copy(tshape)
    M = prod(tshape[rdims])
    N = prod(tshape[cdims])
    ridx = __build_idx(subs, vals, rdims, tshape)
    cidx = __build_idx(subs, vals, cdims, tshape)
    return coo_matrix((vals, (ridx, cidx)), shape=(M, N))


def fold(subs, vals, rdims, tshape, cdims=None):
    if type(rdims) == type(1):
        rdims = [rdims]
    nsubs = zeros((len(vals), len(tshape)), dtype=np.int)
    tshape = array(tshape)
    if cdims is None:
        cdims = setdiff1d(range(len(tshape)), rdims)[::-1]
    if len(rdims) > 0:
        nsubs[:, rdims] = unravel_index(tshape[rdims], subs[0])
    if len(cdims) > 0:
        nsubs[:, cdims] = unravel_index(tshape[cdims], subs[1])
    nsubs = [z.flatten() for z in hsplit(nsubs, len(tshape))]
    return sptensor(nsubs, vals, tshape)


def issparse(obj):
    return isinstance(obj, sptensor)


def __sttm_compute(T, V, mode, transp):
    Z = T.unfold(mode, transp=True).tocsr()
    if transp:
        V = V.T
    Z = Z.dot(V.T)
    shape = copy(T.shape)
    shape[mode] = V.shape[0]
    if issparse_mat(Z):
        newT = fold((Z.row, Z.col), Z.data, [mode], shape)
    else:
        newT = tt.fold(Z.T, mode, shape)
    return newT


def sttm_me_compute(T, V, edims, sdims, transp):
    """
    Assume Y = T x_i V_i for i = 1...n can fit into memory
    """
    shapeY = T.shape.copy()

    # Determine size of Y
    for n in np.union1d(edims, sdims):
        shapeY[n] = V[n].shape[1] if transp else V[n].shape[0]
    print shapeY

    # Allocate Y (final result) and v (vectors for elementwise computations)
    Y = zeros(shapeY)
    shapeY = array(shapeY)
    v = [None for _ in xrange(len(edims))]

    for i in xrange(np.prod(shapeY[edims])):
        print i
        rsubs = unravel_index(shapeY[edims], i)
        print rsubs


def __build_idx(subs, vals, dims, tshape):
    shape = array(tshape[dims], ndmin=1)
    dims = array(dims, ndmin=1)
    if len(shape) == 0:
        idx = ones(len(vals), dtype=vals.dtype)
    elif len(subs) == 0:
        idx = array([])
    else:
        idx = ravel_index([subs[i] for i in dims], shape)
    return idx


def ttv(T, v, dims, vidx, remdims):
    if not isinstance(v, tuple):
        raise ValueError('v must be a tuple of vectors')
    nvals = T.vals
    for n in range(len(dims)):
        idx = T.subs[dims[n]]
        w = v[vidx[n]]
        W = w[idx]
        nvals = nvals * W

    # all dimensions used, return sum
    if len(remdims) == 0:
        return nvals.sum()

    # otherwise accumulate
    nsubs = [T.subs[r] for r in remdims]
    nsz = [T.shape[r] for r in remdims]

    # result is a vector
    if len(remdims) == 1:
        c = accum(nsubs, nvals, nsz)
        if len(np.nonzero(c)[0]) <= 0.5 * nsz:
            return sptensor(arange(nsz), c)
        else:
            return nvals

    # result is an array
    return sptensor(nsubs, nvals, shape=nsz, accumfun=np.sum)

