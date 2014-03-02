"""Magic and hackery.
"""

import json
import theano
import theano.tensor as T

FLOATX = theano.config.floatX

TENSOR_TYPES = {None: T.scalar,
                0: T.vector,
                1: T.matrix,
                2: T.tensor3,
                3: T.tensor4}

from .core import Input
from .core import Output
from .nodes import Affine
from .nodes import Conv3D
from .nodes import Likelihood
from .losses import NegativeLogLikelihood
from .losses import L1Magnitude
from .losses import L2Magnitude
from .framework import Canvas
from .framework import ConnectionManager
from .framework import Graph


def __jsonSupport__():
    """writeme."""
    def encode(self, jsonObject):
        """writeme."""
        return jsonObject.__json__

    def decode(obj):
        """writeme."""
        if 'type' in obj:
            return eval(obj.pop('type')).__json_init__(**obj)
        return obj

    json.JSONEncoder.default = encode
    json._default_decoder = json.JSONDecoder(object_hook=decode)

__jsonSupport__()
