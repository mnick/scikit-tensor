# coding: utf-8
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
"""
This module holds diffent algorithms to compute the CP decomposition, i.e.
algorithms where

.. math:: \\ten{X} \\approx \sum_{r=1}^{rank} \\vec{u}_r^{(1)} \outer \cdots \outer \\vec{u}_r^{(N)}

"""
import logging
import time
import numpy as np
from numpy import array, dot, ones, sqrt
from scipy.linalg import pinv
from numpy.random import rand
from .core import nvecs, norm
from .ktensor import ktensor

_log = logging.getLogger('CP')
_DEF_MAXITER = 500
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-5
_DEF_FIT_METHOD = 'full'
_DEF_TYPE = np.float

__all__ = [
    'als',
    'opt',
    'wopt'
]


def als(X, rank, **kwargs):
    """
    Alternating least-sqaures algorithm to compute the CP decomposition.

    Parameters
    ----------
    X : tensor_mixin
        The tensor to be decomposed.
    rank : int
        Tensor rank of the decomposition.
    init : {'random', 'nvecs'}, optional
        The initialization method to use.
            - random : Factor matrices are initialized randomly.
            - nvecs : Factor matrices are initialzed via HOSVD.
        (default 'nvecs')
    max_iter : int, optional
        Maximium number of iterations of the ALS algorithm.
        (default 500)
    fit_method : {'full', None}
        The method to compute the fit of the factorization
            - 'full' : Compute least-squares fit of the dense approximation of.
                       X and X.
            - None : Do not compute the fit of the factorization, but iterate
                     until ``max_iter`` (Useful for large-scale tensors).
        (default 'full')
    conv : float
        Convergence tolerance on difference of fit between iterations
        (default 1e-5)

    Returns
    -------
    P : ktensor
        Rank ``rank`` factorization of X. ``P.U[i]`` corresponds to the factor
        matrix for the i-th mode. ``P.lambda[i]`` corresponds to the weight
        of the i-th mode.
    fit : float
        Fit of the factorization compared to ``X``
    itr : int
        Number of iterations that were needed until convergence
    exectimes : ndarray of floats
        Time needed for each single iteration

    Examples
    --------
    Create random dense tensor

    >>> from sktensor import dtensor, ktensor
    >>> U = [np.random.rand(i,3) for i in (20, 10, 14)]
    >>> T = dtensor(ktensor(U).toarray())

    Compute rank-3 CP decomposition of ``T`` with ALS

    >>> P, fit, itr, _ = als(T, 3)

    Result is a decomposed tensor stored as a Kruskal operator

    >>> type(P)
    <class 'sktensor.ktensor.ktensor'>

    Factorization should be close to original data

    >>> np.allclose(T, P.totensor())
    True

    References
    ----------
    .. [1] Kolda, T. G. & Bader, B. W.
           Tensor Decompositions and Applications.
           SIAM Rev. 51, 455–500 (2009).
    .. [2] Harshman, R. A.
           Foundations of the PARAFAC procedure: models and conditions for an 'explanatory' multimodal factor analysis.
           UCLA Working Papers in Phonetics 16, (1970).
    .. [3] Carroll, J. D.,  Chang, J. J.
           Analysis of individual differences in multidimensional scaling via an N-way generalization of 'Eckart-Young' decomposition.
           Psychometrika 35, 283–319 (1970).
    """

    # init options
    ainit = kwargs.pop('init', _DEF_INIT)
    maxiter = kwargs.pop('max_iter', _DEF_MAXITER)
    fit_method = kwargs.pop('fit_method', _DEF_FIT_METHOD)
    conv = kwargs.pop('conv', _DEF_CONV)
    dtype = kwargs.pop('dtype', _DEF_TYPE)
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    N = X.ndim
    normX = norm(X)

    U = _init(ainit, X, N, rank, dtype)
    fit = 0
    exectimes = []
    for itr in range(maxiter):
        tic = time.clock()
        fitold = fit

        for n in range(N):
            Unew = X.uttkrp(U, n)
            Y = ones((rank, rank), dtype=dtype)
            for i in (list(range(n)) + list(range(n + 1, N))):
                Y = Y * dot(U[i].T, U[i])
            Unew = Unew.dot(pinv(Y))
            # Normalize
            if itr == 0:
                lmbda = sqrt((Unew ** 2).sum(axis=0))
            else:
                lmbda = Unew.max(axis=0)
                lmbda[lmbda < 1] = 1
            U[n] = Unew / lmbda

        P = ktensor(U, lmbda)
        if fit_method == 'full':
            normresidual = normX ** 2 + P.norm() ** 2 - 2 * P.innerprod(X)
            fit = 1 - (normresidual / normX ** 2)
        else:
            fit = itr
        fitchange = abs(fitold - fit)
        exectimes.append(time.clock() - tic)
        _log.debug(
            '[%3d] fit: %.5f | delta: %7.1e | secs: %.5f' %
            (itr, fit, fitchange, exectimes[-1])
        )
        if itr > 0 and fitchange < conv:
            break

    return P, fit, itr, array(exectimes)


def opt(X, rank, **kwargs):
    ainit = kwargs.pop('init', _DEF_INIT)
    maxiter = kwargs.pop('maxIter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)
    dtype = kwargs.pop('dtype', _DEF_TYPE)
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    N = X.ndim
    U = _init(ainit, X, N, rank, dtype)


def wopt(X, rank, **kwargs):
    raise NotImplementedError()


def _init(init, X, N, rank, dtype):
    """
    Initialization for CP models
    """
    Uinit = [None for _ in range(N)]
    if isinstance(init, list):
        Uinit = init
    elif init == 'random':
        for n in range(1, N):
            Uinit[n] = array(rand(X.shape[n], rank), dtype=dtype)
    elif init == 'nvecs':
        for n in range(1, N):
            Uinit[n] = array(nvecs(X, n, rank), dtype=dtype)
    else:
        raise 'Unknown option (init=%s)' % str(init)
    return Uinit

# vim: set et:
