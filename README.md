scikit-tensor
=============

scikit-tensor is a Python module for multilinear algebra and tensor 
factorizations. Currently, scikit-tensor supports basic tensor operations 
such as folding/unfolding, tensor-matrix and tensor-vector products as 
well as the following tensor factorizations:

* Canonical / Parafac Decomposition
* Tucker Decomposition
* RESCAL
* DEDICOM 
* INDSCAL 

Moreover, all operations support dense and tensors.

Dependencies
------------
The required dependencies to build the software are `Numpy >= 1.3`, `SciPy >= 0.7`.

Usage
-----
Example script to decompose sensory bread data (available from http://www.models.life.ku.dk/datasets) using CP-ALS

```python
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
```

Install
-------
This package uses distutils, which is the default way of installing python modules. To install in your home directory, use

    python setup.py install --user

To install for all users on Unix/Linux

    python setup.py build
    sudo python setup.py install

To install in development mode

    python setup.py develop

Contributing & Development
--------------------------
scikit-tensor is still an extremely young project, and I'm happy for any contributions (patches, code, bugfixes, *documentation*, whatever) to get it to a stable and useful point. Feel free to get in touch with me via email (mnick at AT mit DOT edu) or directly via github.

Development is synchronized via git. To clone this repository, run

    git clone git://github.com/scikit-learn/scikit-learn.git

Authors
-------
Maximilian Nickel <mnick AT mit DOT edu>

+ <http://twitter.com/mnick>
+ <http://github.com/mnick>

License
-------
scikit-tensor is licensed under the GPLv3 <http://www.gnu.org/licenses/gpl-3.0.txt>
