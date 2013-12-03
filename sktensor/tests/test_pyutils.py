from sktensor.pyutils import *
from nose.tools import assert_equal, assert_true, raises

def test_from_to_without():
    frm, to, without = 2, 88, 47
    lst = list(range(frm, without)) + list(range(without + 1, to))
    assert_equal(lst, from_to_without(frm, to, without))

    rlst = list(range(to - 1, without, -1)) + list(range(without - 1, frm - 1,-1))
    assert_equal(rlst, from_to_without(frm, to, without, reverse=True))
    assert_equal(lst[::-1], from_to_without(frm, to, without, reverse=True))
