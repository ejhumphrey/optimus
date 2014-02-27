"""writeme."""

import numpy as np
import theano

from . import primitives
from . import TENSOR_TYPES
from . import FLOATX


class Symbolic(primitives.JObject):
    """Contains common methods/properties/attributes"""
    def __init__(self):
        raise NotImplementedError("Base class! Subclass only.")

    def __str__(self):
        return "<%s: %s>" % (self.otype, self.name)

    @property
    def ndim(self):
        """writeme."""
        return len(self.shape)

    @property
    def name(self):
        return self.variable.name

    @name.setter
    def name(self, name):
        self.variable.name = name


class Input(Symbolic):

    """writeme.

    shape = ...
        None -> scalar
        [] -> vector
        [1, ] -> matrix
        [1, 2, ] -> tensor3
        [1, 2, 3, ] -> tensor4

    """
    def __init__(self, shape):
        self.shape = shape
        self.variable = TENSOR_TYPES[self.ndim](dtype=FLOATX)

    @property
    def __json__(self):
        return dict(otype=self.otype, shape=self.shape)


class Output(Symbolic):
    """writeme."""
    def __init__(self):
        self.shape = []
        self.variable = None

    @property
    def __json__(self):
        """TODO(ejhumphrey@nyu.edu): Serialize shape?"""
        return dict(otype=self.otype)


class Parameter(Symbolic):
    """writeme.

    Note: Include datatype?
    """
    def __init__(self, shape, value=None):
        """writeme."""
        self.shape = shape
        if value is None:
            value = np.zeros(self.shape)
        self.variable = theano.shared(value=value)

    @property
    def __json__(self):
        return dict(shape=self.shape, otype=self.otype)

    @property
    def value(self):
        """writeme."""
        return self.variable.get_value()

    @value.setter
    def value(self, value):
        """writeme."""
        self.variable.set_value(value)
