"""
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

import logging
import time
import numpy as np
from numpy import array, dot, ones, sqrt, diag
from scipy.linalg import pinv
from numpy.random import rand
from core import mttkrp, ktensor, nvecs, norm

_log = logging.getLogger('CP-ALS')
__DEF_MAXITER = 500
__DEF_INIT = 'nvecs'
__DEF_CONV = 1e-7
__DEF_FIT_METHOD = 'full'


def cp_als(X, rank, dtype=np.float32, **kwargs):
    #X = array(X, dtype=dtype)
    N = len(X.shape)
    normX = norm(X)

    # init options
    ainit = kwargs.pop('init', __DEF_INIT)
    maxiter = kwargs.pop('maxIter', __DEF_MAXITER)
    fit_method = kwargs.pop('fit_method', __DEF_FIT_METHOD)
    conv = kwargs.pop('conv', __DEF_CONV)
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    U = __init(ainit, X, N, rank, dtype)
    fit = 0
    exectimes = []
    for itr in xrange(maxiter):
        tic = time.clock()
        fitold = fit

        for n in range(N):
            Unew = mttkrp(X, U, n)
            Y = ones((rank, rank), dtype=dtype)
            for i in (range(n) + range(n + 1, N)):
                Y = Y * dot(U[i].T, U[i])
            Unew = dot(Unew, pinv(Y))
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


def __init(init, X, N, rank, dtype):
    Uinit = [None]
    if isinstance(init, list):
        Uinit = init
    elif init == 'random':
        for n in range(1, N):
            Uinit.append(array(rand(X.shape[n], rank), dtype=dtype))
    elif init == 'nvecs':
        for n in range(1, N):
            Uinit.append(array(nvecs(X, n, rank), dtype=dtype))
    else:
        raise 'Unknown option (init=%s)' % str(init)
    return Uinit
