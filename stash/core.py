"""Core Data Stash Objects


The primary object is an Entity, which consists of named Fields. For
example, an Entity might correspond to some observation, and has three
Fields; chroma, mfccs, and label. Each of these Fields will have a 'value',
which returns the data (scalar, string, list, or array).

The goal of Entities / Fields is to provide a seamless de/serialization
data structure for scaling well with potentially massive datasets.
"""

import numpy as np
import h5py
import json

from . import keyutils


class Entity(object):
    """Struct-like object for getting named fields into and out of a Stash.

    Much like a native Python dictionary, the keyword arguments of an Entity
    will become keys of the object. Additionally, and more importantly, these
    keys are also named attributes:

    >>> x = Entity(a=3, b=5)
    >>> x['a'].value == x.b.value
    False

    See tests/test_stash.py for more examples.
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

    def __delitem__(self, key):
        """writeme."""
        del self.__dict__[key]

    def __len__(self):
        return len(self.keys())

    @classmethod
    def from_hdf5_group(cls, group):
        """writeme."""
        new_grp = cls()
        for key in group:
            new_grp.__dict__[key] = _LazyField(group[key])
        return new_grp

    @property
    def values(self):
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

    @value.setter
    def value(self, value):
        """writeme."""
        self._value = value
        return

    @property
    def attrs(self):
        """writeme."""
        if self._attrs is None:
            self._attrs = dict(self._dataset.attrs)
        return self._attrs


class Stash(h5py.File):
    """On-disk dictionary-like object."""
    __KEYMAP__ = "__KEYMAP__"
    __WIDTH__ = 256
    __DEPTH__ = 3

    def __init__(self, name, mode=None, entity=None, **kwds):
        """
        Parameters
        ----------
        name: str
            Path to file on disk
        mode: str
            Filemode for the object.
        entity: Entity subclass
            Entity class for interpreting objects in the stash.
        """
        h5py.File.__init__(self, name=name, mode=mode, **kwds)

        if entity is None:
            entity = Entity
        self._entity_cls = entity

        self._keymap = self.__decode_keymap__()
        self._agu = keyutils.uniform_hexgen(self.__DEPTH__, self.__WIDTH__)

    def __decode_keymap__(self):
        keymap_array = np.array(h5py.File.get(self, self.__KEYMAP__, '{}'))
        return json.loads(keymap_array.tostring())

    def __encode_keymap__(self, keymap):
        return np.array(json.dumps(keymap))

    def __del__(self):
        """Safe default destructor"""
        if self.fid.valid:
            self.close()

    def close(self):
        """write keys and paths to disk"""
        if self.__KEYMAP__ in self:
            del self[self.__KEYMAP__]
        keymap_str = self.__encode_keymap__(self._keymap)
        h5py.File.create_dataset(
            self, name=self.__KEYMAP__, data=keymap_str)

        h5py.File.close(self)

    def get(self, key):
        """Fetch the entity for a given key."""
        addr = self._keymap.get(key)
        raw_group = h5py.File.get(self, addr)
        raw_key = raw_group.attrs.get("key")
        assert raw_key == key, \
            "Key inconsistency: received '%s', expected '%s'" % (raw_key, key)
        return self._entity_cls.from_hdf5_group(raw_group)

    def add(self, key, entity, overwrite=False):
        """Add a key-entity pair to the File.

        Parameters
        ----------
        key: str
            Key to write the value under.
        entity: Entity
            Object to write to file.
        overwrite: bool, default=False
            Overwrite the key-entity pair if the key currently exists.
        """
        key = str(key)
        if key in self._keymap:
            if not overwrite:
                raise ValueError(
                    "Data exists for '%s'; did you mean overwrite=True?" % key)
            else:
                addr = self.remove(key)
        else:
            addr = self._agu.next()

        while addr in self:
            addr = self._agu.next()

        self._keymap[key] = addr
        new_grp = self.create_group(addr)
        new_grp.attrs.create(name='key', data=key)
        for dset_key, dset in dict(**entity).iteritems():
            new_dset = new_grp.create_dataset(name=dset_key, data=dset.value)
            for k, v in new_dset.attrs.iteritems():
                dset.attrs.create(name=k, data=v)

    def remove(self, key):
        """Delete a key-entity pair from the stash.

        Parameters
        ----------
        key: str
            Key to remove from the Stash.

        Returns
        -------
        address: str
            Absolute internal address freed in the process.
        """
        addr = self._keymap.pop(key, None)
        if addr is None:
            raise ValueError("The key '%s' does not exist." % key)

        del self[addr]
        return addr

    def keys(self):
        """Return a list of all keys in the Stash."""
        return self._keymap.keys()

    def __paths__(self):
        """Return a list of all absolute archive paths in the Stash."""
        return self._keymap.values()

    def __len__(self):
        return len(self.keys())
