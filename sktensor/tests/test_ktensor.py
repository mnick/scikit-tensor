from numpy.random import randn
from sktensor import ktensor


def test_vectorization():
    rank = 5
    shape = (5, 27, 3, 13)
    U = [randn(s, rank) for s in shape]
    K = ktensor(U)
    v = K.tovec()
    K2 = v.toktensor()

    assert sum([s * rank for s in shape]) == len(v.v)
    assert K == K2
