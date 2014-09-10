"""Magic and hackery."""

import json
import theano
import theano.tensor as T

FLOATX = theano.config.floatX

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
from .nodes import Accumulate
from .nodes import Concatenate
from .nodes import Stack

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
from .nodes import RectifiedLinear
from .nodes import Sigmoid
from .nodes import Softmax
from .nodes import Tanh

# Unary Nodes, with params
from .nodes import Affine
from .nodes import Conv3D
from .nodes import Gain
from .nodes import RadialBasis

# Binary Nodes, no params
from .nodes import CrossProduct
from .nodes import SelectIndex
from .nodes import MaxNotIndex
from .nodes import MinNotIndex
from .nodes import SquaredEuclidean
from .nodes import Multiply

# Old losses ... deprecated, don't use.
from .losses import NegativeLogLikelihood
from .losses import MultiNegativeLogLikelihood
from .losses import ConditionalNegativeLogLikelihood
from .losses import MeanSquaredError
from .losses import SparseMeanSquaredError
from .losses import CrossEntropy
from .losses import SparseCrossEntropy
from .losses import Margin
from .losses import ClassificationError

# Framework classes
from .framework import ConnectionManager
from .framework import Graph
from .framework import Driver
from .framework import save
from .framework import load
from .framework import random_init


def __str_convert(obj):
    """Convert unicode to strings.

    Known issue: Uses dictionary comprehension, and is incompatible with 2.6.
    """
    if isinstance(obj, dict):
        return {__str_convert(key): __str_convert(value)
                for key, value in obj.iteritems()}
    elif isinstance(obj, list):
        return [__str_convert(element) for element in obj]
    elif isinstance(obj, unicode):
        return obj.encode('utf-8')
    else:
        return obj


def __jsonSupport__():
    """writeme."""
    def encode(self, jsonObject):
        """writeme."""
        return jsonObject.__json__

    def decode(obj):
        """writeme."""
        # TODO(ejhumphrey): Consider filtering on underscores OR reserved word.
        # filt_obj = dict()
        # for k in obj:
        #     if k.startswith("_"):
        #         continue
        #     filt_obj[k] = obj[k]
        obj = __str_convert(obj)
        if 'type' in obj:
            return eval(obj.pop('type')).__json_init__(**obj)
        return obj

    json.JSONEncoder.default = encode
    json._default_decoder = json.JSONDecoder(object_hook=decode)

__jsonSupport__()
