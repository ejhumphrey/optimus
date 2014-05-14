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

# from . import core
# from . import nodes
from .core import Input
from .core import Output
from .core import Parameter
from .core import Port

from .nodes import Affine
from .nodes import Conv3D
from .nodes import Softmax
from .nodes import MultiSoftmax

from .losses import NegativeLogLikelihood
from .losses import MultiNegativeLogLikelihood
from .losses import L1Magnitude
from .losses import L2Magnitude
from .losses import MeanSquaredError
from .losses import CategoricalCrossEntropy

from .framework import ConnectionManager
from .framework import Graph
from .framework import Driver
from .framework import save
from .framework import load
from .framework import random_init


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
        if 'type' in obj:
            return eval(obj.pop('type')).__json_init__(**obj)
        return obj

    json.JSONEncoder.default = encode
    json._default_decoder = json.JSONDecoder(object_hook=decode)

__jsonSupport__()
