"""
"""

import h5py
from collections import OrderedDict
import numpy as np

from . import core
from . import keyutils
from . import selectors


class File(h5py.File):
    """writeme."""
    __KEYS__ = "__KEYS__"
    __ADDR__ = "__ADDR__"
    __WIDTH__ = 256
    __DEPTH__ = 3

    def __init__(self, name, mode=None, entity=None, **kwds):
        """writeme."""
        h5py.File.__init__(self, name=name, mode=mode, **kwds)

        if entity is None:
            self._entity_cls = core.Entity

        keys = list(h5py.File.get(self, self.__KEYS__, []))
        addrs = list(h5py.File.get(self, self.__ADDR__, []))
        self._key_map = OrderedDict([(k, a) for k, a in zip(keys, addrs)])
        self._agu = keyutils.uniform_hexgen(self.__DEPTH__, self.__WIDTH__)

    def close(self):
        """write keys and paths to disk"""
        if self.__KEYS__ in self:
            del self[self.__KEYS__]
        h5py.File.create_dataset(
            self, name=self.__KEYS__, data=self._key_map.keys())

        if self.__ADDR__ in self:
            del self[self.__ADDR__]
        h5py.File.create_dataset(
            self, name=self.__ADDR__, data=self._key_map.values())
        h5py.File.close(self)

    def get(self, key):
        """
        Fetch the Group for a given key.
        """
        addr = self._key_map.get(key)
        raw_group = h5py.File.get(self, addr)
        raw_key = raw_group.attrs.get("key")
        assert raw_key == key, \
            "Key inconsistency: received '%s', expected '%s'" % (raw_key, key)
        return self._entity_cls.from_hdf5_group(raw_group)

    def add(self, key, entity):
        """writeme."""
        key = str(key)
        assert not key in self._key_map
        addr = self._agu.next()
        while addr in self:
            addr = self._agu.next()

        # print "%s -> %s" % (key, addr)

        self._key_map[key] = addr
        new_grp = self.create_group(addr)
        new_grp.attrs.create(name='key', data=key)
        for dset_key, dset in dict(**entity).iteritems():
            new_dset = new_grp.create_dataset(name=dset_key, data=dset.value)
            for k, v in new_dset.attrs.iteritems():
                dset.attrs.create(name=k, data=v)

    def keys(self):
        """writeme."""
        return self._key_map.keys()

    def paths(self):
        """writeme."""
        return self._key_map.values()

    def __len__(self):
        return len(self.keys())
