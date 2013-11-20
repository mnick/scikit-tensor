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
from numpy import dot, ones, zeros, diag, kron, outer, array, prod, eye
from numpy.linalg import norm, solve, eigvals
from numpy.random import rand
from scipy.linalg import qr
from scipy.sparse.linalg import eigsh
from scipy.optimize import fmin_l_bfgs_b, fmin_ncg, fmin_tnc
from scipy.sparse import issparse

_DEF_MAXITER = 500
_DEF_INIT = 'nvecs'
_DEF_PROJ = True
_DEF_CONV = 1e-5
_DEF_NNE = -1
_DEF_OPTFUNC = 'lbfgs'

_log = logging.getLogger('DEDICOM')
np.seterr(invalid='raise')


def asalsan(X, rank, **kwargs):
    """
    ASALSAN algorithm to compute the three-way DEDICOM decomposition
    of a tensor

    See
    ---
    .. [1] Brett W. Bader, Richard A. Harshman, Tamara G. Kolda
       "Temporal analysis of semantic graphs using ASALSAN"
       7th International Conference on Data Mining, 2007

    .. [2] Brett W. Bader, Richard A. Harshman, Tamara G. Kolda
       "Temporal analysis of Social Networks using Three-way DEDICOM"
       Technical Report, 2006
    """
    # init options
    ainit = kwargs.pop('init', _DEF_INIT)
    proj = kwargs.pop('proj', _DEF_PROJ)
    maxIter = kwargs.pop('maxIter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)
    nne = kwargs.pop('nne', _DEF_NNE)
    optfunc = kwargs.pop('optfunc', _DEF_OPTFUNC)
    if not len(kwargs) == 0:
        raise BaseException('Unknown keywords (%s)' % (kwargs.keys()))

    # init starting points
    D = ones((len(X), rank))
    sz = X[0].shape
    n = sz[0]
    R = rand(rank, rank)
    if ainit == 'random':
        A = rand(n, rank)
    elif ainit == 'nvecs':
        S = zeros((n, n))
        T = zeros((n, n))
        for i in range(len(X)):
            T = X[i]
            S = S + T + T.T
        evals, A = eigsh(S, rank)
        if nne > 0:
            A[A < 0] = 0
        if proj:
            Q, A2 = qr(A)
            X2 = __projectSlices(X, Q)
            R = __updateR(X2, A2, D, R, nne)
        else:
            R = __updateR(X, A, D, R, nne)
    elif isinstance(ainit, np.ndarray):
        A = ainit
    else:
        raise 'Unknown init option ("%s")' % ainit

    # perform decomposition
    if issparse(X[0]):
        normX = [norm(M.data) ** 2 for M in X]
        Xflat = [M.tolil().reshape((1, prod(M.shape))).tocsr() for M in X]
    else:
        normX = [norm(M) ** 2 for M in X]
        Xflat = [M.flatten() for M in X]
    M = zeros((n, n))
    normXSum = sum(normX)
    #normX = norm(X)**2
    fit = fitold = f = fitchange = 0
    exectimes = []
    for iters in xrange(maxIter):
        tic = time.clock()
        fitold = fit
        A = __updateA(X, A, D, R, nne)
        if proj:
            Q, A2 = qr(A)
            X2 = __projectSlices(X, Q)
            R = __updateR(X2, A2, D, R, nne)
            D, f = __updateD(X2, A2, D, R, nne, optfunc)
        else:
            R = __updateR(X, A, D, R, nne)
            D, f = __updateD(X, A, D, R, nne, optfunc)

        # compute fit
        f = 0
        for i in xrange(len(X)):
            AD = dot(A, diag(D[i, :]))
            M = dot(dot(AD, R), AD.T)
            f += normX[i] + norm(M) ** 2 - 2 * Xflat[i].dot(M.flatten())
        f *= 0.5
        fit = 1 - (f / normXSum)
        fitchange = abs(fitold - fit)

        exectimes.append(time.clock() - tic)

        # print iter info when debugging is enabled
        _log.debug('[%3d] fit: %.5f | delta: %7.1e | secs: %.5f' % (
            iters, fit, fitchange, exectimes[-1]
        ))

        if iters > 1 and fitchange < conv:
            break
    return A, R, D, fit, iters, array(exectimes)


def __updateA(X, A, D, R, nne):
    rank = A.shape[1]
    F = zeros((X[0].shape[0], rank))
    E = zeros((rank, rank))

    AtA = dot(A.T, A)
    for i in range(len(X)):
        Dk = diag(D[i, :])
        DRD = dot(Dk, dot(R, Dk))
        DRtD = DRD.T
        F += X[i].dot(dot(A, DRtD)) + X[i].T.dot(dot(A, DRD))
        E += dot(DRD, dot(AtA, DRtD)) + dot(DRtD, dot(AtA, DRD))
    if nne > 0:
        E = dot(A, E) + nne
        A = A * (F / E)
    else:
        A = solve(E.T, F.T).T
    return A


def __updateR(X, A, D, R, nne):
    r = A.shape[1] ** 2
    T = zeros((r, r))
    t = zeros(r)
    for i in range(len(X)):
        AD = dot(A, diag(D[i, :]))
        ADt = AD.T
        tmp = dot(ADt, AD)
        T = T + kron(tmp, tmp)
        tmp = dot(ADt, X[i].dot(AD))
        t = t + tmp.flatten()
    r = A.shape[1]
    if nne > 0:
        Rflat = R.flatten()
        T = dot(T, Rflat) + nne
        R = (Rflat * t / T).reshape(r, r)
    else:
        # TODO check if this is correct
        R = solve(T, t).reshape(r, r)
        #R = (pinv(T + eye(r ** 2)).dot(t)).reshape(r, r)
    return R


def __updateD(X, A, D, R, nne, optfunc):
    f = 0
    for i in range(len(X)):
        d = D[i, :]
        u = Updater(X[i], A, R)
        if nne > 0:
            bounds = len(d) * [(0, None)]
            res = fmin_l_bfgs_b(
                u.updateD_F, d, u.updateD_G, factr=1e12, bounds=bounds
            )
        else:
            if optfunc == 'lbfgs':
                res = fmin_l_bfgs_b(u.updateD_F, d, u.updateD_G, factr=1e12)
                D[i, :] = res[0]
                f += res[1]
            elif optfunc == 'ncg':
                res = fmin_ncg(
                    u.updateD_F, d, u.updateD_G, fhess=u.updateD_H,
                    full_output=True, disp=False
                )
                # TODO: check return value of ncg and update D, f
                raise NotImplementedError()
            elif optfunc == 'tnc':
                res = fmin_tnc(u.updateD_F, d, u.updateD_G, disp=False)
                # TODO: check return value of tnc and update D, f
                raise NotImplementedError()
    return D, f


class Updater:
    def __init__(self, Z, A, R):
        self.Z = Z
        self.A = A
        self.R = R
        self.x = None

    def precompute(self, x, cache=True):
        if not cache or self.x is None or (x != self.x).any():
            self.AD = dot(self.A, diag(x))
            self.ADt = self.AD.T
            self.E = self.Z - dot(self.AD, dot(self.R, self.ADt))

    def updateD_F(self, x):
        self.precompute(x)
        return norm(self.E, 'fro') ** 2

    def updateD_G(self, x):
        """
        Compute Gradient for update of D

        See [2] for derivation of Gradient
        """
        self.precompute(x)
        g = zeros(len(x))
        Ai = zeros(self.A.shape[0])
        for i in range(len(g)):
            Ai = self.A[:, i]
            g[i] = (self.E * (dot(self.AD, outer(self.R[:, i], Ai)) +
                    dot(outer(Ai, self.R[i, :]), self.ADt))).sum()
        return -2 * g

    def updateD_H(self, x):
        """
        Compute Hessian for update of D

        See [2] for derivation of Hessian
        """
        self.precompute(x)
        H = zeros((len(x), len(x)))
        Ai = zeros(self.A.shape[0])
        Aj = zeros(Ai.shape)
        for i in range(len(x)):
            Ai = self.A[:, i]
            ti = dot(self.AD, outer(self.R[:, i], Ai)) + dot(outer(Ai, self.R[i, :]), self.ADt)

            for j in range(i, len(x)):
                Aj = self.A[:, j]
                tj = outer(Ai, Aj)
                H[i, j] = (
                    self.E * (self.R[i, j] * tj + self.R[j, i] * tj.T) -
                    ti * (
                        dot(self.AD, outer(self.R[:, j], Aj)) +
                        dot(outer(Aj, self.R[j, :]), self.ADt)
                    )
                ).sum()
                H[j, i] = H[i, j]
        H *= -2
        e = eigvals(H).min()
        H = H + (eye(H.shape[0]) * e)
        return H


def __projectSlices(X, Q):
    X2 = []
    for i in range(len(X)):
        X2.append(Q.T.dot(X[i].dot(Q)))
    return X2
