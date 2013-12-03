def inherit_docstring_from(cls):
    def docstring_inheriting_decorator(fn):
        fn.__doc__ = getattr(cls, fn.__name__).__doc__
        return fn
    return docstring_inheriting_decorator


def is_sequence(obj):
    """
    Helper function to determine sequences
    across Python 2.x and 3.x
    """
    try:
        from collections import Sequence
    except ImportError:
        from operator import isSequenceType
        return isSequenceType(obj)
    else:
        return isinstance(obj, Sequence)


def is_number(obj):
    """
    Helper function to determine numbers
    across Python 2.x and 3.x
    """
    try:
        from numbers import Number
    except ImportError:
        from operator import isNumberType
        return isNumberType(obj)
    else:
        return isinstance(obj, Number)


def func_attr(f, attr):
    """
    Helper function to get the attribute of a function
    like, name, code, defaults across Python 2.x and 3.x
    """
    if hasattr(f, 'func_%s' % attr):
        return getattr(f, 'func_%s' % attr)
    elif hasattr(f, '__%s__' % attr):
        return getattr(f, '__%s__' % attr)
    else:
        raise ValueError('Object %s has no attr' % (str(f), attr))


def from_to_without(frm, to, without, step=1, skip=1, reverse=False, separate=False):
    """
    Helper function to create ranges with missing entries
    """
    if reverse:
        frm, to = (to - 1), (frm - 1)
        step *= -1
        skip *= -1
    a = list(range(frm, without, step))
    b = list(range(without + skip, to, step))
    if separate:
        return a, b
    else:
        return a + b
