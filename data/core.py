"""
"""


class NamedStruct(object):
    """writeme."""
    def __init__(self, **kwargs):
        object.__init__(self)
        for key, value in kwargs.iteritems():
            self.add(key, value)

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '%s{%s}' % (self.__class__.__name__, ", ".join(self.keys()))

    def keys(self):
        """writeme."""
        return self.__dict__.keys()

    def add(self, key, value):
        """writeme."""
        self.__dict__[key] = value

    def __getitem__(self, key):
        """writeme."""
        return self.__dict__[key]

    def __len__(self):
        return len(self.keys())


class Feature(object):
    """writeme.
    metadata: similar thoughts, but potentially more important.
    """
    def __init__(self, value, attrs=None):
        """writeme."""

        self.value = value
        if attrs is None:
            attrs = dict()
        self.attrs = attrs

    @classmethod
    def from_hdf5_dataset(cls, hdf5_dataset):
        """This might be poor practice."""
        return LazyFeature(hdf5_dataset)


class LazyFeature(object):
    """Lazy-loading Feature for reading data from HDF5 files.

    Do not use directly."""
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


class Entity(NamedStruct):
    """writeme."""

    @classmethod
    def from_hdf5_group(cls, group):
        """writeme."""
        new_grp = cls()
        for key in group:
            new_grp.add(key, LazyFeature(group[key]))
        return new_grp
