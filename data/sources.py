"""
"""
#import time
import h5py
import json
import numpy as np

from . import core
from . import keyutils


class File(h5py.File):
    """writeme."""
    __KEYMAP__ = "__KEYMAP__"
    __WIDTH__ = 256
    __DEPTH__ = 3

    def __init__(self, name, mode=None, entity=None, **kwds):
        """writeme."""
        h5py.File.__init__(self, name=name, mode=mode, **kwds)

        if entity is None:
            self._entity_cls = core.Entity

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
        """
        Fetch the Group for a given key.
        """
        addr = self._keymap.get(key)
        raw_group = h5py.File.get(self, addr)
        raw_key = raw_group.attrs.get("key")
        assert raw_key == key, \
            "Key inconsistency: received '%s', expected '%s'" % (raw_key, key)
        return self._entity_cls.from_hdf5_group(raw_group)

    def add(self, key, entity):
        """writeme."""
        key = str(key)
        assert not key in self._keymap
        addr = self._agu.next()
        while addr in self:
            addr = self._agu.next()

        # print "%s -> %s" % (key, addr)

        self._keymap[key] = addr
        new_grp = self.create_group(addr)
        new_grp.attrs.create(name='key', data=key)
        for dset_key, dset in dict(**entity).iteritems():
            new_dset = new_grp.create_dataset(name=dset_key, data=dset.value)
            for k, v in new_dset.attrs.iteritems():
                dset.attrs.create(name=k, data=v)

    def keys(self):
        """writeme."""
        return self._keymap.keys()

    def paths(self):
        """writeme."""
        return self._keymap.values()

    def __len__(self):
        return len(self.keys())
