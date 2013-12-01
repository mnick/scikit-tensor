import numpy as np
from numpy import cumprod, array, arange, zeros, floor, lexsort


def accum(subs, vals, func=np.sum, sorted=False, shape=None, with_subs=False):
    """
    NumPy implementation for Matlab's accumarray
    """

    # sort accmap for ediff if not sorted
    if not sorted:
        sidx = lexsort(subs, axis=0)
        subs = [sub[sidx] for sub in subs]
        vals = vals[sidx]
    idx = np.where(np.ediff1d(subs, to_begin=[1], to_end=[1]))[0]

    # create values array
    if shape is None:
        nvals = np.zeros(len(idx) - 1)
    else:
        nvals = np.zeros(shape)
    for i in range(len(idx) - 1):
        nvals[i] = func(vals[idx[i]:idx[i + 1]])

    # return results
    if with_subs:
        return nvals, tuple(sub[idx[:-1]] for sub in subs)
    else:
        return nvals


def unravel_dimension(shape, idx):
    if isinstance(idx, type(1)):
        idx = array([idx])
    k = [1] + list(cumprod(shape[:-1]))
    n = len(shape)
    subs = zeros((len(idx), n), dtype=np.int)
    for i in arange(n - 1, -1, -1):
        subs[:, i] = floor(idx / k[i])
        idx = idx % k[i]
    return subs
