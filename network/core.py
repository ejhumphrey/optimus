"""TODO(ejhumphrey): write me."""

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
        raise NotImplementedError(
            "<%s>: Missing a JSON serialization property." % self.type)

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s>' % self.type

    def keys(self):
        """TODO(ejhumphrey): write me."""
        names = list()
        for k in self.__dict__.keys():
            if k.startswith("_"):
                continue
            names.append(k)
        return names

    def __getitem__(self, key):
        """TODO(ejhumphrey): write me."""
        return self.__dict__[key]

    def __len__(self):
        return len(self.keys())

    @property
    def type(self):
        """TODO(ejhumphrey): write me."""
        return self.__class__.__name__


class Port(object):
    """writeme

    Doesn't do any shape validation so... guess that's more for convenience
    than anything else.
    """
    def __init__(self, name, shape=None):
        self._variable = [None]
        self.shape = shape
        self.name = name

    # TODO(ejhumphrey): Does a port ever needs to be serialized...?
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
        """TODO(ejhumphrey): write me."""
        return self.__class__.__name__

    def reset(self):
        """TODO(ejhumphrey): write me."""
        self._variable[0] = None

    def connect(self, source):
        """TODO(ejhumphrey): write me."""
        self._variable = source._variable
        # Save shape too?

    @property
    def variable(self):
        """TODO(ejhumphrey): write me."""
        return self._variable[0]

    @variable.setter
    def variable(self, value):
        """TODO(ejhumphrey): write me."""
        self._variable[0] = value

    @property
    def ndim(self):
        """TODO(ejhumphrey): write me."""
        if self.shape is None:
            return None
        return len(self.shape)


class PortList(object):
    """writeme

    Doesn't do any shape validation so... guess that's more for convenience
    than anything else.
    """
    def __init__(self, name):
        self._variable = []
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
        """TODO(ejhumphrey): write me."""
        return self.__class__.__name__

    def reset(self):
        """TODO(ejhumphrey): write me."""
        while self._variable:
            self._variable.pop(0)

    def connect(self, source):
        """TODO(ejhumphrey): write me."""
        self._variable.append(source._variable)

    @property
    def variable(self):
        """TODO(ejhumphrey): write me."""
        return self._variable

    # TODO(ejhumphrey): Is this necessary??
    # --------------------------------------
    # @variable.setter
    # def variable(self, value):
    #     """TODO(ejhumphrey): write me."""
    #     self._variable[0] = value

    # @property
    # def ndim(self):
    #     """TODO(ejhumphrey): write me."""
    #     if self.shape is None:
    #         return None
    #     return len(self.shape)


class Input(Port):
    """writeme.

    shape = ...
        None -> scalar
        [1,] -> vector
        [1, 2, ] -> matrix
        [1, 2, 3, ] -> tensor3
        [1, 2, 3, 4, ] -> tensor4

    """
    # TODO(ejhumphrey): Shouldn't the order of these be switched?
    def __init__(self, shape, name, dtype=None):
        self.shape = shape
        # List representation to keep consistency with Ports
        if dtype is None:
            dtype = FLOATX
        self._variable = [TENSOR_TYPES[self.ndim](name=name, dtype=dtype)]

    @property
    def __json__(self):
        """TODO(ejhumphrey): write me."""
        return dict(
            shape=self.shape,
            name=self.name,
            dtype=self.dtype,
            type=self.type)

    @property
    def dtype(self):
        """TODO(ejhumphrey): write me."""
        return self._variable[0].dtype

    @property
    def name(self):
        """TODO(ejhumphrey): write me."""
        return self._variable[0].name

    @name.setter
    def name(self, name):
        """TODO(ejhumphrey): write me."""
        self._variable[0].name = name


class Output(Port):
    """TODO(ejhumphrey): write me."""

    def reset(self):
        """TODO(ejhumphrey): write me."""
        self.shape = []
        self.variable = None


class Parameter(object):
    """writeme.

    Note: Include datatype?
    """
    # TODO(ejhumphrey): Parameters should never be anonymous, right?
    def __init__(self, shape, name="anonymous", value=None):
        """TODO(ejhumphrey): write me."""
        self.shape = shape
        if value is None:
            value = np.zeros(self.shape, dtype=FLOATX)
        self._variable = theano.shared(value=value)
        self.name = name

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s: %s>' % (self.type, self.name)

    @property
    def type(self):
        """TODO(ejhumphrey): write me."""
        return self.__class__.__name__

    @property
    def value(self):
        """TODO(ejhumphrey): write me."""
        return self.variable.get_value()

    @value.setter
    def value(self, value):
        """TODO(ejhumphrey): write me."""
        self.variable.set_value(value.astype(FLOATX))

    @property
    def name(self):
        """TODO(ejhumphrey): write me."""
        return self.variable.name

    @name.setter
    def name(self, name):
        """TODO(ejhumphrey): write me."""
        self.variable.name = name

    @property
    def variable(self):
        """TODO(ejhumphrey): write me."""
        return self._variable
