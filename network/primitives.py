"""Primitive JSON types for safe serialization.

"""

import json


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

    @property
    def otype(self):
        return self.__class__.__name__


class Struct(JObject):
    """Struct object

    This object behaves like a JavaScript object, in that attributes can be
    accessed either by key (like a dict) or self.attr (like a class).
    """
    def __init__(self, **kwargs):
        self.update(kwargs)

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
        """TODO(ejhumphrey@nyu.edu): writeme."""
        return self.__dict__[key]

    def __len__(self):
        return len(self.keys())

    def items(self):
        """writeme."""
        return dict([(k, self[k]) for k in self.keys()])
