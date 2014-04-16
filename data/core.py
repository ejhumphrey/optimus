"""

Data Objects:

The top-level object is an Entity, which consists of named Features. For
example, and Entity might correspond to some observation, and has three
Features; chroma, mfccs, and label. Each of these features will have a 'value'
field, to access the numerical representation contained therein.

The goal of entities / features is to provide a seamless de/serialization
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

    """
    def __init__(self, **kwargs):
        object.__init__(self)
        for key, value in kwargs.iteritems():
            self.add(key, Feature(value))

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '%s{%s}' % (self.__class__.__name__, ", ".join(self.keys()))

    def keys(self):
        """writeme."""
        return self.__dict__.keys()

    def add(self, key, value):
        """TODO(ejhumphrey): Why this instead of __setitem__? Is it an h5py
        consistency thing?

        Parameters
        ----------
        key: str
            Attribute name, must be valid python syntax.
        value: Feature
            Feature corresponding to the given key.
        """
        self.__dict__[key] = value

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
            new_grp.add(key, _LazyFeature(group[key]))
        return new_grp


class Feature(object):
    """Wrapper around array data for optimus."""
    def __init__(self, value, attrs=None):
        """writeme."""
        self.value = np.asarray(value)
        if attrs is None:
            attrs = dict()
        self.attrs = attrs

    @classmethod
    def from_hdf5_dataset(cls, hdf5_dataset):
        """This might be poor practice."""
        return _LazyFeature(hdf5_dataset)


class _LazyFeature(object):
    """Lazy-loading Feature for reading data from HDF5 files.

    Note: Do not use directly. This class provides a common interface with
    Features, but only returns information as needed, wrapping h5py types."""
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
