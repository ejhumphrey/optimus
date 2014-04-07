"""writeme"""


from . import sources
from . import selectors
import numpy as np


class LocalCache(object):
    """Basically, a dictionary with some bonus methods.
    """
    def __init__(self, refresh_prob, selector=None):
        """writeme."""

        if selector is None:
            selector = selectors.permute_items
        self._selector = selector

        self._data = dict()
        self._refresh_prob = refresh_prob
        self._item_gen = self._selector(self._data)

    def get(self, key):
        """
        Fetch the Group for a given key.
        """
        return self._data.get(key)

    def add(self, key, entity):
        """writeme."""
        self._data[key] = entity
        self._item_gen = self._selector(self._data)

    def keys(self):
        """writeme."""
        return self._data.keys()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self.get(key)

    def remove(self, key, prob=1):
        """Swap an existing key-value pair with a new one."""
        # Refresh on success.
        if np.random.binomial(1, p=prob):
            print "Deleting %s" % key
            self.remove(key)

    def next(self):
        """Return the next item."""
        key, entity = self._item_gen.next()
        self.remove(key, prob=self._refresh_prob)
        return key, entity


class Queue(object):
    """
    The Queue Pipeline:

    Source
      |
    Selector
      |
    Transformer
      |
    Buffer


    """
    def __init__(self, source, batch_size, transformers=None, selector=None,
                 cache_size=1000, refresh_prob=0.01):
        """writeme"""
        self._source = source
        self.batch_size = batch_size

        # If None, safely create list
        if transformers is None:
            transformers = list()
        self._transformers = transformers

        # If None, create default selector
        if selector is None:
            selector = selectors.permute_items
        self._selector = selector(self._source)

        # If the requested cache size is larger than the number of items in
        # the source, snap it down and disable auto-refresh.
        if cache_size > len(source):
            cache_size = len(source)
            refresh_prob = 0

        self.cache = LocalCache(refresh_prob=refresh_prob, selector=selector)
        self._cache_size = cache_size
        self.populate()
        self.update()

    def populate(self):
        """writeme"""
        while len(self.cache) <= self._cache_size:
            key, entity = self._selector.next()
            self.cache.add(key, entity)

    def update(self):
        """writeme"""
        self.clear()
        self.populate()
        while not self.ready:
            self.process()

    # --- Subclassable methods ---
    def process(self):
        """writeme"""
        entity = self.cache.next()[-1]
        for fx in self._transformers:
            entity = fx(entity)
        self.buffer(entity)

    def buffer(self, entity):
        """writeme"""
        raise NotImplementedError("Write me")

    def clear(self):
        """writeme"""
        raise NotImplementedError("Write me")

    @property
    def ready(self):
        """writeme"""
        raise NotImplementedError("Write my stopping condition")
