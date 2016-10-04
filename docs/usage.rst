Usage
=====

Example script to decompose sensory bread data (available from http://www.models.life.ku.dk/datasets) using CP-ALS

.. code:: python

    import logging
    from scipy.io.matlab import loadmat
    from sktensor import dtensor, cp_als

    # Set logging to DEBUG to see CP-ALS information
    logging.basicConfig(level=logging.DEBUG)

    # Load Matlab data and convert it to dense tensor format
    mat = loadmat('../data/sensory-bread/brod.mat')
    T = dtensor(mat['X'])

    # Decompose tensor using CP-ALS
    P, fit, itr, exectimes = cp_als(T, 3, init='random')
