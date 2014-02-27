"""writeme."""

import numpy as np
import theano

import json
from . import FLOATX
from . import TENSOR_TYPES


def _jsonSupport():
    """TODO(ejhumphrey@nyu.edu): writeme."""
    def default(self, jsonObject):
        return jsonObject.__json__

    json.JSONEncoder.default = default
    json._default_decoder = json.JSONDecoder()

_jsonSupport()


class JObject(object):
    @property
    def __json__(self):
        raise NotImplementedError("Write this property for JSON support.")

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s>' % self.otype

    @property
    def otype(self):
        """writeme."""
        return self.__class__.__name__


class Struct(JObject):
    """Struct object

    This object behaves like a JavaScript object, in that attributes can be
    accessed either by key (like a dict) or self.attr (like a class).
    """
    def __init__(self, **kwargs):
        self.update(**kwargs)

    @property
    def __json__(self):
        return self.items()

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s>' % self.otype

    def keys(self):
        """writeme."""
        keys = list()
        for k in self.__dict__.keys():
            if k.startswith("_"):
                continue
            keys.append(k)
        return keys

    def update(self, **kwargs):
        """writeme."""
        for name, value in kwargs.iteritems():
            self.__dict__[name] = value

    def __getitem__(self, key):
        """writeme."""
        return self.__dict__[key]

    def __len__(self):
        return len(self.keys())

    def items(self):
        """writeme."""
        return dict([(k, self[k]) for k in self.keys()])


class Port(JObject):
    """writeme"""
    def __init__(self, shape=None, name="anonymous"):
        self._variable = [None]
        self.shape = shape
        self.name = name

    @property
    def __json__(self):
        return dict(shape=self.shape)

    def reset(self):
        """writeme"""
        self._variable[0] = None

    def connect(self, source):
        """writeme"""
        self._variable = source._variable

    @property
    def variable(self):
        """writeme"""
        return self._variable[0]

    @variable.setter
    def variable(self, value):
        """writeme"""
        self._variable[0] = value

    @property
    def ndim(self):
        """writeme."""
        return len(self.shape)


class Input(Port):
    """writeme.

    shape = ...
        None -> scalar
        [] -> vector
        [1, ] -> matrix
        [1, 2, ] -> tensor3
        [1, 2, 3, ] -> tensor4

    """
    def __init__(self, shape, name):
        self.shape = shape
        self._variable = [TENSOR_TYPES[self.ndim](name=name, dtype=FLOATX)]

    @property
    def __json__(self):
        return dict(name=self.name, shape=self.shape)

    @property
    def name(self):
        """writeme."""
        return self._variable[0].name

    @name.setter
    def name(self, name):
        """writeme."""
        self._variable[0].name = name


class Output(Port):
    """writeme."""

    def reset(self):
        self.shape = []
        self.variable = None


class Parameter(JObject):
    """writeme.

    Note: Include datatype?
    """
    def __init__(self, shape, name="anonymous", value=None):
        """writeme."""
        self.shape = shape
        if value is None:
            value = np.zeros(self.shape, dtype=FLOATX)
        self.variable = theano.shared(value=value)
        self.name = name

    @property
    def __json__(self):
        return dict(shape=self.shape, otype=self.otype)

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s: %s>' % (self.otype, self.name)

    @property
    def value(self):
        """writeme."""
        return self.variable.get_value()

    @value.setter
    def value(self, value):
        """writeme."""
        self.variable.set_value(value.astype(FLOATX))

    @property
    def name(self):
        """writeme."""
        return self.variable.name

    @name.setter
    def name(self, name):
        """writeme."""
        self.variable.name = name
