"""Core Data Objects


The main object is an Entity, which consists of named Fields. For
example, an Entity might correspond to some observation, and has three
Fields; chroma, mfccs, and label. Each of these Fields will have a 'value',
which returns the data (scalar, string, list, or array).

The goal of Entities / Fields is to provide a seamless de/serialization
data structure for scaling well with potentially massive datasets.
"""

import numpy as np


class Entity(object):
    """Struct-like object for getting named fields into and out of Files.

    Much like a native Python dictionary, the keyword arguments of an Entity
    will become keys of the object. Additionally, and more importantly, these
    keys are also named attributes:

    >>> x = Entity(a=3, b=5)
    >>> x['a'].value == x.b.value
    False

    See tests/test_data.py for more examples.
    """
    def __init__(self, **kwargs):
        object.__init__(self)
        for key, value in kwargs.iteritems():
            self[key] = value

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '%s{%s}' % (self.__class__.__name__, ", ".join(self.keys()))

    def keys(self):
        """writeme."""
        return self.__dict__.keys()

    def __setitem__(self, key, value):
        """Set value for key.

        Parameters
        ----------
        key: str
            Attribute name, must be valid python attribute syntax.
        value: scalar, string, list, or np.ndarray
            Data corresponding to the given key.
        """
        self.__dict__[key] = Field(value)

    def __getitem__(self, key):
        """writeme."""
        return self.__dict__[key]

    def __len__(self):
        return len(self.keys())

    @classmethod
    def from_hdf5_group(cls, group):
        """writeme."""
        new_grp = cls()
        for key in group:
            new_grp.__dict__[key] = _LazyField(group[key])
        return new_grp

    def todict(self):
        return dict([(k, self[k].value) for k in self.keys()])


class Field(object):
    """Data wrapper.

    You should seldom, if ever, need to create Fields explicitly.
    """
    def __init__(self, value, attrs=None):
        """writeme."""
        self.value = np.asarray(value)
        if attrs is None:
            attrs = dict()
        self.attrs = attrs

    @classmethod
    def from_hdf5_dataset(cls, hdf5_dataset):
        """This might be poor practice."""
        return _LazyField(hdf5_dataset)


class _LazyField(object):
    """Lazy-loading Field for reading data from HDF5 files.

    Note: Do not use directly. This class provides a common interface with
    Fields, but only returns information as needed, wrapping h5py types."""
    def __init__(self, hdf5_dataset):
        self._dataset = hdf5_dataset
        self._value = None
        self._attrs = None

    @property
    def value(self):
        """writeme."""
        if self._value is None:
            self._value = self._dataset.value
        return self._value

    @property
    def attrs(self):
        """writeme."""
        if self._attrs is None:
            self._attrs = dict(self._dataset.attrs)
        return self._attrs
