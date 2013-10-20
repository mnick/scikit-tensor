"""
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
from numpy import array, dot, outer, zeros, ones, arange, kron
from numpy import setdiff1d
from scipy.linalg import eigh
from scipy.sparse import issparse as issparse_mat
from scipy.sparse.linalg import eigsh
from operator import isSequenceType
from abc import ABCMeta, abstractmethod
#from coremod import khatrirao


class tensor_mixin(object):

    __metaclass__ = ABCMeta

    def ttm(self, V, mode=None, transp=False, without=False):
        """
        Tensor times matrix product

        Parameter
        ---------
        T:

        >>> T = zeros((3, 4, 2))
        >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
        >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
        >>> V = array([[1, 3, 5], [2, 4, 6]])

        >>> Y = ttm(T, V, 0)
        >>> Y[:, :, 0]
        array([[  22.,   49.,   76.,  103.],
            [  28.,   64.,  100.,  136.]])

        >>> Y[:, :, 1]
        array([[ 130.,  157.,  184.,  211.],
            [ 172.,  208.,  244.,  280.]])

        """
        if mode is None:
            mode = range(self.ndim)
        if isinstance(V, np.ndarray):
            Y = self._ttm_compute(V, mode, transp)
        elif isSequenceType(V):
            dims, vidx = check_multiplication_dims(mode, self.ndim, len(V), vidx=True, without=without)
            Y = self._ttm_compute(V[vidx[0]], dims[0], transp)
            for i in xrange(1, len(dims)):
                Y = self.ttm_compute(Y, V[vidx[i]], dims[i], transp)
        return Y

        def ttv(self, v, dims=[]):
            """
            Tensor times vector product
        
            Parameter
            ---------
            """
            if not isinstance(v, tuple):
                v = (v, )
            dims, vidx = check_multiplication_dims(dims, self.ndim, len(v), vidx=True)
            for i in range(len(dims)):
                if not len(v[vidx[i]]) == self.shape[dims[i]]:
                    raise ValueError('Multiplicant is wrong size')
            remdims = np.setdiff1d(range(self.ndim), dims)
            return self._ttv_compute(v, dims, vidx, remdims)

        @abstractmethod
        def _ttm_compute(self, V, mode, transp):
            pass


def __ttm_single(self, V, mode, transp):
    """
    Helper function, dispatches ttm to sparse or dense tensor impementation
    """
    if issparse(T):
        return stt.__sttm_compute(T, V, mode, transp)
    elif isinstance(T, tensor):
        return T.ttm(V, mode, transp)
    elif isinstance(T, np.ndarray):
        return tensor(T).ttm(V, mode, transp)




def check_multiplication_dims(dims, N, M, vidx=False, without=False):
    dims = array(dims, ndmin=1)
    if len(dims) == 0:
        dims = arange(N)
    if without:
        dims = setdiff1d(range(N), dims)
    if not np.in1d(dims, arange(N)).all():
        raise ValueError('Invalid dimensions')
    P = len(dims)
    sidx = np.argsort(dims)
    sdims = dims[sidx]
    if vidx:
        if M > N:
            raise ValueError('More multiplicants than dimensions')
        if M != N and M != P:
            raise ValueError('Invalid number of multiplicants')
        if P == M:
            vidx = sidx
        else:
            vidx = sdims
        return sdims, vidx
    else:
        return sdims


def unfold(X, n):
    """
    Unfold X in mode n

    >>> T = zeros((3, 4, 2))
    >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
    >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]

    >>> unfold(T, 0)
    array([[  1.,   4.,   7.,  10.,  13.,  16.,  19.,  22.],
           [  2.,   5.,   8.,  11.,  14.,  17.,  20.,  23.],
           [  3.,   6.,   9.,  12.,  15.,  18.,  21.,  24.]])

    >>> unfold(T, 1)
    array([[  1.,   2.,   3.,  13.,  14.,  15.],
           [  4.,   5.,   6.,  16.,  17.,  18.],
           [  7.,   8.,   9.,  19.,  20.,  21.],
           [ 10.,  11.,  12.,  22.,  23.,  24.]])

    >>> unfold(T, 2)
    array([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,
             12.],
           [ 13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,  22.,  23.,
             24.]])
    """
    if issparse(X):
        return X.unfold([n]).tocsr()
    elif isinstance(X, tt.tensor):
        return X.unfold(n)
    elif isinstance(X, np.ndarray):
        return unfold(tt.tensor(X), n)
    else:
        raise ValueError('Unsupported object (%s)' % type(X))


def fold(X, n, shape):
    if issparse(X):
        return X.fold(n, shape)
    elif isinstance(X, np.ndarray):
        return unfolded_tensor(X, n, shape).fold()


def transpose(X, axes=None):
    if issparse(X):
        return stt.transpose(X, axes=axes)
    else:
        return np.transpose(X, axes=axes)


def concatenate(tpl, axis=None):
    if issparse(tpl[0]):
        return stt.concatenate(tpl, axis)
    else:
        return np.concatenate(tpl, axis)


def innerprod(X, Y):
    """
    inner prodcut with a Tensor
    """
    return dot(X.flatten(), Y.flatten())


def uttkrp(X, U, n):
    "Unfolded tensor x Khatri-Rao product"
    order = range(n) + range(n + 1, X.ndim)
    Z = khatrirao([U[i] for i in order], reverse=True)
    return X.unfold(n).dot(Z)


def nvecs(X, n, rank, do_flipsign=True):
    """
    Eigendecomposition of mode-n unfolding of a tensor
    """
    Xn = unfold(X, n)
    Y = Xn.dot(Xn.T)
    if issparse_mat(Y):
        _, U = eigsh(Y, rank, which='LM')
    else:
        N = Y.shape[0]
        _, U = eigh(Y, eigvals=(N - rank, N - 1))
        #_, U = eigsh(Y, rank, which='LM')
    # reverse order of eigenvectors such that eigenvalues are decreasing
    U = array(U[:, ::-1])
    # flip sign
    if do_flipsign:
        U = flipsign(U)
    return U


def norm(X):
    """
    Frobenius norm for tensors

    See
    [Kolda and Bader, 2009; p.457]
    """
    if issparse(X):
        return np.linalg.norm(X.vals)
    else:
        return np.linalg.norm(X.flatten())


def flipsign(U):
    """
    Flip sign of factor matrices such that largest magnitude
    element will be positive
    """
    midx = abs(U).argmax(axis=0)
    for i in xrange(U.shape[1]):
        if U[midx[i], i] < 0:
            U[:, i] = -U[:, i]
    return U


def center(X, n):
    Xn = unfold(X, n)
    N = Xn.shape[0]
    m = Xn.sum(axis=0) / N
    m = kron(m, ones((N, 1)))
    Xn = Xn - m
    return fold(Xn, n)


def center_matrix(X):
    m = X.mean(axis=0)
    return X - m


def scale(X, n):
    Xn = unfold(X, n)
    m = np.float_(np.sqrt((Xn ** 2).sum(axis=1)))
    m[m == 0] = 1
    for i in range(Xn.shape[0]):
        Xn[i, :] = Xn[i] / m[i]
    return fold(Xn, n, X.shape)


# TODO more efficient cython implementation
def khatrirao(A, reverse=False):
    """
    Khatri-Rao product
    """
    N = A[0].shape[1]
    M = 1
    for i in range(len(A)):
        if A[i].ndim != 2:
            raise ValueError('A must be a list of matrices')
        elif N != A[i].shape[1]:
            raise ValueError('All matrices must have same number of columns')
        M *= A[i].shape[0]
    matorder = arange(len(A))
    if reverse:
        matorder = matorder[::-1]
    # preallocate
    P = np.zeros((M, N), dtype=A[0].dtype)
    for n in range(N):
        ab = A[matorder[0]][:, n]
        for j in range(1, len(matorder)):
            ab = np.kron(A[matorder[j]][:, n], ab)
        P[:, n] = ab
    return P


class ktensor(object):

    def __init__(self, U, lmbda=None):
        self.U = U
        self.shape = [Ui.shape[0] for Ui in U]
        self.rank = U[0].shape[1]
        self.lmbda = lmbda
        if not all(array([Ui.shape[1] for Ui in U]) == self.rank):
            raise ValueError('Dimension mismatch of factor matrices')
        if lmbda is None:
            self.lmbda = ones(self.rank)

    def mttkrp(self, U, n):
        N = len(self.shape)
        if n == 1:
            R = U[1].shape[1]
        else:
            R = U[0].shape[1]
        W = np.tile(self.lmbda, 1, R)
        for i in range(n) + range(n + 1, N):
            W = W * dot(self.U[i].T, U[i])
        return dot(self.U[n], W)

    def norm(self):
        N = len(self.shape)
        coef = outer(self.lmbda, self.lmbda)
        for i in range(N):
            coef = coef * dot(self.U[i].T, self.U[i])
        return np.sqrt(coef.sum())

    def innerprod(self, X):
        N = len(self.shape)
        R = len(self.lmbda)
        res = 0
        for r in range(R):
            vecs = []
            for n in range(N):
                vecs.append(self.U[n][:, r])
            res += self.lmbda[r] * ttv(X, tuple(vecs))
        return res

    def toarray(self):
        A = dot(self.lmbda, khatrirao(self.U, reverse=True).T)
        return A.reshape(self.shape)


def teneye(dim, order):
    """
    Create tensor with superdiagonal all one, rest zeros
    """
    I = zeros(dim ** order)
    for f in range(dim):
        idd = f
        for i in range(1, order):
            idd = idd + dim ** (i - 1) * (f - 1)
        I[idd] = 1
    return I.reshape(ones(order) * dim)


def tvecmat(m, n):
    d = m * n
    i2 = arange(d).reshape(m, n).T.flatten()
    Tmn = zeros((d, d))
    Tmn[arange(d), i2] = 1
    return Tmn

    #i = arange(d);
    #rI = m * (i-1)-(m*n-1) * floor((i-1)/n)
    #print rI
    #I1s = s2i((d,d), rI, arange(d))
    #print I1s
    #Tmn[I1s] = 1
    #return Tmn.reshape((d,d)).T
