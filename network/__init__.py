"""Magic and hackery.
"""

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
from .nodes import Likelihood

from .losses import NegativeLogLikelihood
from .losses import L1Magnitude
from .losses import L2Magnitude

from .framework import ConnectionManager
from .framework import Graph
from .framework import Driver


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
