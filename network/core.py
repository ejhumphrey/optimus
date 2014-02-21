"""writeme."""

import json
import numpy as np
import theano.tensor as T
import theano


# def _jsonSupport():
#     """writeme."""
#     def default(self, xObject):
#         """writeme."""
#         return dict([(k, xObject[k]) for k in xObject])

#     json.JSONEncoder.default = default
#     json._default_decoder = json.JSONDecoder()

# _jsonSupport()


# class Struct(object):
#     """Dict-like object for JSON Serialization.

#     This object behaves like a dictionary to allow init-level attribute names,
#     seamless JSON-serialization, and double-star style unpacking (**obj).

#     To hide an attribute from serialization, make the variable 'private' with
#     a leading underscore and expose it via the '@property' decorator.
#     """
#     def __init__(self, **kwargs):
#         object.__init__(self)
#         for name, value in kwargs.iteritems():
#             self.__dict__[name] = value

#     def __repr__(self):
#         """Render the object as an unambiguous string."""
#         return '<%s>' % self.__class__.__name__

#     def _jsonSupport():
#         """writeme."""
#         def default(self, xObject):
#             return dict([(k, xObject[k]) for k in xObject])

#         json.JSONEncoder.default = default
#         json._default_decoder = json.JSONDecoder()

#     _jsonSupport()

#     def keys(self):
#         """writeme."""
#         keys = list()
#         for k in self.__dict__.keys():
#             if k.startswith("_"):
#                 continue
#             keys.append(k)
#         return keys

#     # def update(self, **kwargs):
#     #     """writeme."""
#     #     for name, value in kwargs.iteritems():
#     #         self.__dict__[name] = value

#     def __getitem__(self, key):
#         """writeme."""
#         return self.__dict__[key]

#     def __len__(self):
#         return len(self.keys())

class Struct(object):
    """Dict-like object for JSON Serialization.

    This object behaves like a dictionary to allow init-level attribute names,
    seamless JSON-serialization, and double-star style unpacking (**obj).
    """
    def __init__(self, **kwargs):
        object.__init__(self)
        for name, value in kwargs.iteritems():
            self.__dict__[name] = value

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s>' % self.__class__.__name__

    def _jsonSupport(*args):
        """TODO(ejhumphrey@nyu.edu): writeme."""
        def default(self, xObject):
            result = dict()
            for k in xObject.keys():
                result[k] = xObject.__dict__[k]
            return result

        json.JSONEncoder.default = default
        json._default_decoder = json.JSONDecoder()

    _jsonSupport()

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
        """TODO(ejhumphrey@nyu.edu): writeme."""
        return self.__dict__[key]

    def __len__(self):
        return len(self.keys())


class Value(Struct):
    """writeme."""
    def __init__(self, value):
        """writeme."""
        self.value = value


class Port(Struct):
    """writeme."""
    def __init__(self, shape):
        """writeme."""
        self.shape = shape
        self._variable = None

    @property
    def ndim(self):
        """writeme"""
        return len(self.shape)

    @property
    def variable(self):
        """writeme"""
        return self._variable

    @variable.setter
    def variable(self, variable):
        """writeme."""
        self._variable = variable


class Parameter(Struct):
    """writeme."""
    def __init__(self, shape):
        """writeme."""
        self.shape = shape
        self._variable = theano.shared(value=np.zeros(self.shape))

    @property
    def variable(self):
        """writeme."""
        return self._variable

    @property
    def value(self):
        """writeme."""
        return self.variable.get_value()

    @value.setter
    def value(self, value):
        """writeme."""
        self.variable.set_value(value)

    @property
    def ndim(self):
        """writeme."""
        return len(self.shape)

    def set_name(self, name):
        """writeme."""
        self._variable.name = name


class Scalar(Struct):
    """writeme."""
    def __init__(self):
        """writeme."""
        self._variable = T.scalar()

    @property
    def variable(self):
        """writeme."""
        return self._variable

    def set_name(self, name):
        """writeme."""
        self._variable.name = name
