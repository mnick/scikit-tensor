import numpy as np
from numpy import cumprod, array, arange, zeros, floor, dot, argsort
from operator import isNumberType


def accum(subs, vals, func=np.sum, sorted=False, sz=None, with_subs=False):
    """
    NumPy implementation for Matlab's accumarray
    """
    # sort accmap for ediff of not sorted
    if not sorted:
        sidx = argsort(subs, axis=0)
        subs = subs[sidx]
        vals = vals[sidx]
    idx = np.where(np.ediff1d(subs, to_begin=[1], to_end=[1]))[0]

    # create values array
    if sz is None:
        nvals = np.zeros(len(idx) - 1)
    else:
        nvals = np.zeros(sz)
    for i in xrange(len(idx) - 1):
        nvals[i] = func(vals[idx[i]:idx[i + 1]])

    # return results
    if with_subs:
        return nvals, subs[idx[:-1]]
    else:
        return nvals


def ravel_index(subs, shape):
    subs = array(subs).T
    mult = [1] + list(cumprod(array(shape)[:-1]))
    return dot(subs, array(mult).T)


def unravel_index(shape, idx):
    if isNumberType(idx):
        idx = array([idx])
    k = [1] + list(cumprod(shape[:-1]))
    n = len(shape)
    subs = zeros((len(idx), n), dtype=np.int)
    for i in arange(n - 1, -1, -1):
        subs[:, i] = floor(idx / k[i])
        idx = idx % k[i]
    return subs
