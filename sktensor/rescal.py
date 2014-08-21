# coding: utf-8
# rescal.py - python script to compute the RESCAL tensor factorization
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

import logging
import time
import numpy as np
from numpy import dot, zeros, array, eye, kron, prod
from numpy.linalg import norm, solve, inv, svd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from numpy.random import rand

__version__ = "0.5"
__all__ = ['als']

_DEF_MAXITER = 100
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-4
_DEF_LMBDA = 0
_DEF_ATTR = []
_DEF_NO_FIT = 1e9
_DEF_FIT_METHOD = None

_log = logging.getLogger('RESCAL')


def als(X, rank, **kwargs):
    """
    RESCAL-ALS algorithm to compute the RESCAL tensor factorization.


    Parameters
    ----------
    X : list
        List of frontal slices X_k of the tensor X.
        The shape of each X_k is ('N', 'N').
        X_k's are expected to be instances of scipy.sparse.csr_matrix
    rank : int
        Rank of the factorization
    lmbdaA : float, optional
        Regularization parameter for A factor matrix. 0 by default
    lmbdaR : float, optional
        Regularization parameter for R_k factor matrices. 0 by default
    lmbdaV : float, optional
        Regularization parameter for V_l factor matrices. 0 by default
    attr : list, optional
        List of sparse ('N', 'L_l') attribute matrices. 'L_l' may be different
        for each attribute
    init : string, optional
        Initialization method of the factor matrices. 'nvecs' (default)
        initializes A based on the eigenvectors of X. 'random' initializes
        the factor matrices randomly.
    compute_fit : boolean, optional
        If true, compute the fit of the factorization compared to X.
        For large sparse tensors this should be turned of. None by default.
    maxIter : int, optional
        Maximium number of iterations of the ALS algorithm. 500 by default.
    conv : float, optional
        Stop when residual of factorization is less than conv. 1e-5 by default

    Returns
    -------
    A : ndarray
        array of shape ('N', 'rank') corresponding to the factor matrix A
    R : list
        list of 'M' arrays of shape ('rank', 'rank') corresponding to the
        factor matrices R_k
    fval : float
        function value of the factorization
    itr : int
        number of iterations until convergence
    exectimes : ndarray
        execution times to compute the updates in each iteration

    Examples
    --------
    >>> X1 = csr_matrix(([1,1,1], ([2,1,3], [0,2,3])), shape=(4, 4))
    >>> X2 = csr_matrix(([1,1,1,1], ([0,2,3,3], [0,1,2,3])), shape=(4, 4))
    >>> A, R, fval, iter, exectimes = rescal([X1, X2], 2)

    See
    ---
    For a full description of the algorithm see:
    .. [1] Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
        "A Three-Way Model for Collective Learning on Multi-Relational Data",
        ICML 2011, Bellevue, WA, USA

    .. [2] Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
        "Factorizing YAGO: Scalable Machine Learning for Linked Data"
        WWW 2012, Lyon, France
    """

    # ------------ init options ----------------------------------------------
    ainit = kwargs.pop('init', _DEF_INIT)
    maxIter = kwargs.pop('maxIter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)
    lmbdaA = kwargs.pop('lambda_A', _DEF_LMBDA)
    lmbdaR = kwargs.pop('lambda_R', _DEF_LMBDA)
    lmbdaV = kwargs.pop('lambda_V', _DEF_LMBDA)
    func_compute_fval = kwargs.pop('compute_fval', _DEF_FIT_METHOD)
    orthogonalize = kwargs.pop('orthogonalize', False)
    P = kwargs.pop('attr', _DEF_ATTR)
    dtype = kwargs.pop('dtype', np.float)

    # ------------- check input ----------------------------------------------
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    # check frontal slices have same size and are matrices
    sz = X[0].shape
    for i in range(len(X)):
        if X[i].ndim != 2:
            raise ValueError('Frontal slices of X must be matrices')
        if X[i].shape != sz:
            raise ValueError('Frontal slices of X must be all of same shape')
        #if not issparse(X[i]):
            #raise ValueError('X[%d] is not a sparse matrix' % i)

    if func_compute_fval is None:
        if orthogonalize:
            func_compute_fval = _compute_fval_orth
        elif prod(X[0].shape) * len(X) > _DEF_NO_FIT:
            _log.warn('For large tensors automatic computation of fit is disabled by default\nTo compute the fit, call rescal.als with "compute_fit=True"\nPlease note that this might cause memory and runtime problems')
            func_compute_fval = None
        else:
            func_compute_fval = _compute_fval

    n = sz[0]
    k = len(X)

    _log.debug(
        '[Config] rank: %d | maxIter: %d | conv: %7.1e | lmbda: %7.1e' %
        (rank, maxIter, conv, lmbdaA)
    )
    _log.debug('[Config] dtype: %s / %s' % (dtype, X[0].dtype))

    # ------- convert X and P to CSR ------------------------------------------
    for i in range(k):
        if issparse(X[i]):
            X[i] = X[i].tocsr()
            X[i].sort_indices()
    for i in range(len(P)):
        if issparse(P[i]):
            P[i] = P[i].tocoo().tocsr()
            P[i].sort_indices()

    # ---------- initialize A ------------------------------------------------
    _log.debug('Initializing A')
    if ainit == 'random':
        A = array(rand(n, rank), dtype=dtype)
    elif ainit == 'nvecs':
        S = csr_matrix((n, n), dtype=dtype)
        for i in range(k):
            S = S + X[i]
            S = S + X[i].T
        _, A = eigsh(csr_matrix(S, dtype=dtype, shape=(n, n)), rank)
        A = array(A, dtype=dtype)
    else:
        raise ValueError('Unknown init option ("%s")' % ainit)

    # ------- initialize R and Z ---------------------------------------------
    R = _updateR(X, A, lmbdaR)
    Z = _updateZ(A, P, lmbdaV)

    # precompute norms of X
    normX = [sum(M.data ** 2) for M in X]

    #  ------ compute factorization ------------------------------------------
    fit = fitchange = fitold = f = 0
    exectimes = []
    for itr in range(maxIter):
        tic = time.time()
        fitold = fit
        A = _updateA(X, A, R, P, Z, lmbdaA, orthogonalize)
        R = _updateR(X, A, lmbdaR)
        Z = _updateZ(A, P, lmbdaV)

        # compute fit value
        if func_compute_fval is not None:
            fit = func_compute_fval(X, A, R, P, Z, lmbdaA, lmbdaR, lmbdaV, normX)
        else:
            fit = np.Inf

        fitchange = abs(fitold - fit)

        toc = time.time()
        exectimes.append(toc - tic)

        _log.debug('[%3d] fval: %0.5f | delta: %7.1e | secs: %.5f' % (
            itr, fit, fitchange, exectimes[-1]
        ))
        if itr > 0 and fitchange < conv:
            break
    return A, R, f, itr + 1, array(exectimes)


# ------------------ Update A ------------------------------------------------
def _updateA(X, A, R, P, Z, lmbdaA, orthogonalize):
    """Update step for A"""
    n, rank = A.shape
    F = zeros((n, rank), dtype=A.dtype)
    E = zeros((rank, rank), dtype=A.dtype)

    AtA = dot(A.T, A)

    for i in range(len(X)):
        F += X[i].dot(dot(A, R[i].T)) + X[i].T.dot(dot(A, R[i]))
        E += dot(R[i], dot(AtA, R[i].T)) + dot(R[i].T, dot(AtA, R[i]))

    # regularization
    I = lmbdaA * eye(rank, dtype=A.dtype)

    # attributes
    for i in range(len(Z)):
        F += P[i].dot(Z[i].T)
        E += dot(Z[i], Z[i].T)

    # finally compute update for A
    A = solve(I + E.T, F.T).T
    return orth(A) if orthogonalize else A


# ------------------ Update R ------------------------------------------------
def _updateR(X, A, lmbdaR):
    rank = A.shape[1]
    U, S, Vt = svd(A, full_matrices=False)
    Shat = kron(S, S)
    Shat = (Shat / (Shat ** 2 + lmbdaR)).reshape(rank, rank)
    R = []
    for i in range(len(X)):
        Rn = Shat * dot(U.T, X[i].dot(U))
        Rn = dot(Vt.T, dot(Rn, Vt))
        R.append(Rn)
    return R


# ------------------ Update Z ------------------------------------------------
def _updateZ(A, P, lmbdaZ):
    Z = []
    if len(P) == 0:
        return Z
    #_log.debug('Updating Z (Norm EQ, %d)' % len(P))
    pinvAt = inv(dot(A.T, A) + lmbdaZ * eye(A.shape[1], dtype=A.dtype))
    pinvAt = dot(pinvAt, A.T).T
    for i in range(len(P)):
        if issparse(P[i]):
            Zn = P[i].tocoo().T.tocsr().dot(pinvAt).T
        else:
            Zn = dot(pinvAt.T, P[i])
        Z.append(Zn)
    return Z


def _compute_fval(X, A, R, P, Z, lmbdaA, lmbdaR, lmbdaZ, normX):
    """Compute fit for full slices"""
    f = lmbdaA * norm(A) ** 2
    for i in range(len(X)):
        ARAt = dot(A, dot(R[i], A.T))
        f += (norm(X[i] - ARAt) ** 2) / normX[i] + lmbdaR * norm(R[i]) ** 2
    return f


def _compute_fval_orth(X, A, R, P, Z, lmbdaA, lmbdaR, lmbdaZ, normX):
    f = lmbdaA * norm(A) ** 2
    for i in range(len(X)):
        f += (normX[i] - norm(R[i]) ** 2) / normX[i] + lmbdaR * norm(R[i]) ** 2
    return f


def sptensor_to_list(X):
    from scipy.sparse import lil_matrix
    if X.ndim != 3:
        raise ValueError('Only third-order tensors are supported (ndim=%d)' % X.ndim)
    if X.shape[0] != X.shape[1]:
        raise ValueError('First and second mode must be of identical length')
    N = X.shape[0]
    K = X.shape[2]
    res = [lil_matrix((N, N)) for _ in range(K)]
    for n in range(X.nnz()):
        res[X.subs[2][n]][X.subs[0][n], X.subs[1][n]] = X.vals[n]
    return res

def orth(A):
    [U, _, Vt] = svd(A, full_matrices=0)
    return dot(U, Vt)
