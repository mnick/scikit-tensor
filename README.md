# scikit-tensor
![Travis CI](https://travis-ci.org/mnick/scikit-tensor.svg?branch=master)

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

#### Dependencies
The required dependencies to build the software are `Numpy >= 1.3`, `SciPy >= 0.7`.

#### Usage
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

#### Install
This package uses distutils, which is the default way of installing python modules. The use of virtual environments is recommended.

    pip install scikit-tensor

To install in development mode

    git clone git@github.com:mnick/scikit-tensor.git
    pip install -e scikit-tensor/

#### Contributing & Development
scikit-tensor is still an extremely young project, and I'm happy for any contributions (patches, code, bugfixes, *documentation*, whatever) to get it to a stable and useful point. Feel free to get in touch with me via email (mnick at AT mit DOT edu) or directly via github.

Development is synchronized via git. To clone this repository, run

    git clone git://github.com/mnick/scikit-tensor.git

#### Authors
Maximilian Nickel: [Web](http://web.mit.edu/~mnick/www), [Email](mailto://mnick AT mit DOT edu), [Twitter](http://twitter.com/mnick)

#### License
scikit-tensor is licensed under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.txt)

#### Related Projects
* [Matlab Tensor Toolbox](http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.5.html): 
  A Matlab toolbox for tensor factorizations and tensor operations freely available for research and evaluation.
* [Matlab Tensorlab](http://www.tensorlab.net/)
  A Matlab toolbox for tensor factorizations, complex optimization, and tensor optimization freely available for
  non-commercial academic research.
