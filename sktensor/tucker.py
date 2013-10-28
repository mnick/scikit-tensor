# sktensor.tucker - Algorithms to compute Tucker decompositions
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
from numpy import array, ones, sqrt
from numpy.random import rand
from operator import isNumberType
from core import ttm, nvecs, norm

__all__ = [
    'hooi',
    'hosvd',
]

_log = logging.getLogger('TUCKER')
__DEF_MAXITER = 500
__DEF_INIT = 'nvecs'
__DEF_CONV = 1e-7


def hooi(X, rank, **kwargs):
    """
    Compute Tucker decomposition of a tensor using Higher-Order Orthogonal
    Iterations.

    Parameters
    ----------
    X : tensor_mixin
        The tensor to be decomposed
    rank : array_like
        The rank of the decomposition for each mode of the tensor.
        The length of ``rank`` must match the number of modes of ``X``.
    init : {'random', 'nvecs'}, optional
        The initialization method to use.
            - random : Factor matrices are initialized randomly.
            - nvecs : Factor matrices are initialzed via HOSVD.
        default : 'nvecs'

    Examples
    --------
    Create dense tensor

    >>> T = np.zeros((3, 4, 2))
    >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
    >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
    >>> T = dtensor(T)

    Compute Tucker decomposition of ``T`` with n-rank [2, 3, 1] via higher-order
    orthogonal iterations

    >>> Y = hooi(T, [2, 3, 1], init='nvecs')

    Shape of the core tensor matches n-rank of the decomposition.

    >>> Y['core'].shape
    (2, 3, 1)
    >>> Y['U'][1].shape
    (3, 2)

    References
    ----------
    .. [1] L. De Lathauwer, B. De Moor, J. Vandewalle: On the best rank-1 and
           rank-(R_1, R_2, \ldots, R_N) approximation of higher order tensors;
           IEEE Trans. Signal Process. 49 (2001), pp. 2262-2271
    """
    # init options
    ainit = kwargs.pop('init', __DEF_INIT)
    maxIter = kwargs.pop('maxIter', __DEF_MAXITER)
    conv = kwargs.pop('conv', __DEF_CONV)
    dtype = kwargs.pop('dtype', X.dtype)
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    ndims = X.ndim
    if isNumberType(rank):
        rank = rank * ones(ndims)

    normX = norm(X)

    U = __init(ainit, X, ndims, rank, dtype)
    fit = 0
    exectimes = []
    for itr in xrange(maxIter):
        tic = time.clock()
        fitold = fit

        for n in range(ndims):
            Utilde = ttm(X, U, n, transp=True, without=True)
            U[n] = nvecs(Utilde, n, rank[n])

        # compute core tensor to get fit
        core = ttm(Utilde, U, n, transp=True)

        # since factors are orthonormal, compute fit on core tensor
        normresidual = sqrt(normX ** 2 - norm(core) ** 2)

        # fraction explained by model
        fit = 1 - (normresidual / normX)
        fitchange = abs(fitold - fit)
        exectimes.append(time.clock() - tic)

        _log.debug(
            '[%3d] fit: %.5f | delta: %7.1e | secs: %.5f'
            % (itr, fit, fitchange, exectimes[-1])
        )
        if itr > 1 and fitchange < conv:
            break
    return core, U

def hosvd(X, rank, dims=None, dtype=None, compute_core=True):
    U = [None for _ in xrange(X.ndim)]
    if dims is None:
        dims = range(X.ndim)
    if dtype is None:
        dtype = X.dtype
    for d in dims:
        U[d] = array(nvecs(X, d, rank[d]), dtype=dtype)
    if compute_core:
        core = X.ttm(U, transp=True)
        return U, core
    else:
        return U

def __init(init, X, N, rank, dtype):
    # Don't compute initial factor for first index, gets computed in
    # first iteration
    Uinit = [None]
    if isinstance(init, list):
        Uinit = init
    elif init == 'random':
        for n in range(1, N):
            Uinit.append(array(rand(X.shape[n], rank[n]), dtype=dtype))
    elif init == 'nvecs':
        Uinit = hosvd(X, rank, range(1, N), dtype=dtype, compute_core=False)
    return Uinit
