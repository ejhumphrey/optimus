"""Magic and hackery."""

import json

import theano
import theano.tensor as T

FLOATX = theano.config.floatX
__OBJECT_TYPE__ = 'type'
TENSOR_TYPES = {None: T.scalar,
                1: T.vector,
                2: T.matrix,
                3: T.tensor3,
                4: T.tensor4}

# Core objects
from .core import Input
from .core import Output
from .core import Parameter
from .core import Port

# Multi-Input Nodes
from .nodes import Add
from .nodes import Concatenate
from .nodes import Stack

# No inputs, single param
from .nodes import Constant

# Unary Nodes, no params, dim-redux
from .nodes import Dimshuffle
from .nodes import Flatten
from .nodes import Slice
from .nodes import L1Magnitude
from .nodes import L2Magnitude
from .nodes import Max
from .nodes import Mean
from .nodes import Min
from .nodes import Sum

# Unary Nodes, no params
from .nodes import Log
from .nodes import Normalize
from .nodes import NormalizeDim
from .nodes import Power
from .nodes import RectifiedLinear
from .nodes import SoftRectifiedLinear
from .nodes import Sigmoid
from .nodes import Sqrt
from .nodes import Softmax
from .nodes import Tanh

# Unary Nodes, with params
from .nodes import Affine
from .nodes import CenteredAffine
from .nodes import Conv3D
from .nodes import Multiply
from .nodes import RadialBasis

# Binary Nodes, no params
from .nodes import CrossProduct
from .nodes import SelectIndex
from .nodes import MaxNotIndex
from .nodes import MinNotIndex
from .nodes import Euclidean
from .nodes import SquaredEuclidean
from .nodes import Divide
from .nodes import Product

# Scalar losses
from .losses import NegativeLogLikelihood
from .losses import MeanSquaredError
from .losses import CrossEntropy
from .losses import CrossEntropyLoss
from .losses import WeightDecayPenalty
from .losses import SimilarityMargin
from .losses import ContrastiveMargin
from .losses import PairwiseRank

# Framework classes
from .framework import ConnectionManager
from .framework import Graph
from .framework import Driver

# Misc Functionality
from .util import array_stepper
from .util import concatenate_data
from .util import random_init

from .version import version as __version__


def __str_convert__(obj):
    """Convert unicode to strings.

    Known issue: Uses dictionary comprehension, and is incompatible with 2.6.
    """
    if isinstance(obj, dict):
        return {__str_convert__(key): __str_convert__(value)
                for key, value in obj.iteritems()}
    elif isinstance(obj, list):
        return [__str_convert__(item) for item in obj]
    elif isinstance(obj, unicode):
        return obj.encode('utf-8')
    else:
        return obj


class JSONSupport():
    """Context manager for temporary JSON support."""
    def __enter__(self):
        # Encoder returns the object's `__json__` property.
        json.JSONEncoder.default = lambda self, jobj: jobj.__json__

        # Decoder looks for the class name, and calls it's class constructor.
        def decode(obj):
            if __OBJECT_TYPE__ in obj:
                return eval(obj.pop(__OBJECT_TYPE__)).__json_init__(**obj)
            return obj

        json._default_decoder = json.JSONDecoder(object_hook=decode)
        return

    def __exit__(self, type, value, traceback):
        # Nothing to do here...
        pass


def save(graph, def_file, param_file=None):
    """Save a graph to disk."""
    if param_file:
        graph.save_param_values(param_file)

    with JSONSupport():
        with open(def_file, "w") as fp:
            json.dump(graph, fp, indent=2)


def load(def_file, param_file=None):
    """Load a graph and corresponding parameter values from disk."""
    with JSONSupport():
        graph = json.load(open(def_file))

    if param_file:
        graph.load_param_values(param_file)
    return graph
