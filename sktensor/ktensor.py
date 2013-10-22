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

import numpy as np
from numpy import dot, ones, array, outer
from sktensor.core import khatrirao


class ktensor(object):

    def __init__(self, U, lmbda=None):
        self.U = U
        self.shape = tuple(Ui.shape[0] for Ui in U)
        self.rank = U[0].shape[1]
        self.lmbda = lmbda
        if not all(array([Ui.shape[1] for Ui in U]) == self.rank):
            raise ValueError('Dimension mismatch of factor matrices')
        if lmbda is None:
            self.lmbda = ones(self.rank)

    def uttkrp(self, U, n):

        """
        Unfolded tensor times Khatri-Rao product for Kruskal tensors

        Parameters
        ----------
        X: tensor_mixin
        U: list of array-like
        n: int

        See also
        --------
        For efficient computations unfolded tensor times Khatri-Rao products
        for other tensors see also
        - sptensor.uttkrp
        - ttensor.uttkrp

        References
        ----------
        [1] B.W. Bader, T.G. Kolda
            Efficient Matlab Computations With Sparse and Factored Tensors
            SIAM J. Sci. Comput, Vol 30, No. 1, pp. 205--231, 2007
        """
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
            res += self.lmbda[r] * X.ttv(tuple(vecs))
        return res

    def toarray(self):
        A = dot(self.lmbda, khatrirao(tuple(self.U)).T)
        return A.reshape(self.shape)

# vim: set et:
