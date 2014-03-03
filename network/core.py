"""writeme."""

import numpy as np
import theano

from . import FLOATX
from . import TENSOR_TYPES


class JObject(object):
    """Base JSON object for complex Optimus types.

    This class offers several key pieces of functionality:
    - JSON Encoding through the __json__ property
    - JSON Decoding through the __json_init__ classmethod
    - both obj[attr] and obj.attr style attribute access
    - **unpacking magic (May not need this anymore?)
    """
    @classmethod
    def __json_init__(cls, **kwargs):
        return cls(**kwargs)

    @property
    def __json__(self):
        raise NotImplementedError("Missing a JSON serialization property.")

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s>' % self.type

    def keys(self):
        """writeme."""
        names = list()
        for k in self.__dict__.keys():
            if k.startswith("_"):
                continue
            names.append(k)
        return names

    def __getitem__(self, key):
        """writeme"""
        return self.__dict__[key]

    def __len__(self):
        return len(self.keys())

    @property
    def type(self):
        """writeme."""
        return self.__class__.__name__


# class Struct(JObject):
#     """Struct object

#     This object behaves like a JavaScript object, in that attributes can be
#     accessed either by key (like a dict) or self.attr (like a class).
#     """
#     def __init__(self, **kwargs):
#         self.update(**kwargs)

#     @property
#     def __json__(self):
#         return self.items()

#     def __repr__(self):
#         """Render the object as an unambiguous string."""
#         return '<%s>' % self.type

#     def keys(self):
#         """writeme."""
#         keys = list()
#         for k in self.__dict__.keys():
#             if k.startswith("_"):
#                 continue
#             keys.append(k)
#         return keys

#     def update(self, **kwargs):
#         """writeme."""
#         for name, value in kwargs.iteritems():
#             self.__dict__[name] = value

#     def __getitem__(self, key):
#         """writeme."""
#         return self.__dict__[key]

#     def __len__(self):
#         return len(self.keys())

#     def items(self):
#         """writeme."""
#         return dict([(k, self[k]) for k in self.keys()])


class Port(object):
    """writeme

    Doesn't do any shape validation so... guess that's more for convenience
    than anything else.
    """
    def __init__(self, name, shape=None):
        self._variable = [None]
        self.shape = shape
        self.name = name

    # I don't know that a port ever needs to be serialized....
    @property
    def __json__(self):
        return dict(name=self.name, type=self.type)

    @classmethod
    def __json_init__(cls, **kwargs):
        return cls(**kwargs)

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s: %s>' % (self.type, self.name)

    @property
    def type(self):
        """writeme."""
        return self.__class__.__name__

    def reset(self):
        """writeme"""
        self._variable[0] = None

    def connect(self, source):
        """writeme"""
        self._variable = source._variable
        # Save shape too?

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
        if self.shape is None:
            return None
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
    def __init__(self, shape, name, dtype=None):
        self.shape = shape
        # List representation to keep consistency with Ports
        if dtype is None:
            dtype = FLOATX
        self._variable = [TENSOR_TYPES[self.ndim](name=name, dtype=dtype)]

    @property
    def __json__(self):
        return dict(
            shape=self.shape,
            name=self.name,
            dtype=self.dtype,
            type=self.type)

    @property
    def dtype(self):
        return self._variable[0].dtype

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


class Parameter(object):
    """writeme.

    Note: Include datatype?
    """
    def __init__(self, shape, name="anonymous", value=None):
        """writeme."""
        self.shape = shape
        if value is None:
            value = np.zeros(self.shape, dtype=FLOATX)
        self._variable = theano.shared(value=value)
        self.name = name

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s: %s>' % (self.type, self.name)

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

    @property
    def variable(self):
        return self._variable
