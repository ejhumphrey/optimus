"""writeme"""


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


def unpack_entities(entities):
    """
    Parameters
    ----------
    entities: list of Entities
        Note that all Entities must have identical fields.

    Returns
    -------
    arrays: dict of np.ndarrays
        Values in 'arrays' are keyed by the
    """
    arrays = dict([(k, []) for k in entities[0].keys()])
    for entity in entities:
        for k in entity.keys():
            if not k in arrays:
                raise ValueError(
                    "All Entities must have the same fields: %s" % entity)
            arrays[k].append(entity[k].value)

    for k in arrays:
        arrays[k] = np.asarray(arrays[k])
    return arrays


class Queue(object):
    """
    Note: Could potentially take a list of Feature (field) names, which may be
    a subset of the fields in an entity.
    Alternatively, it'd be just as easy to introduce a Transformer that deletes
    or manages such fields (and then the unpack method never needs to change).

    The BatchQueue Pipeline:

     source
      |
    Selector
      |
    Transformer(s)
      |
    Serializer
      |
     result

    """
    def __init__(self, source, batch_size, transformers=None, selector=None,
                 serializer=None, cache_size=1000, refresh_prob=0.01):
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

        if serializer is None:
            serializer = unpack_entities
        self._serializer = serializer

        # If the requested cache size is larger than the number of items in
        # the source, snap it down and disable auto-refresh.
        if cache_size > len(source):
            cache_size = len(source)
            refresh_prob = 0

        self.cache = LocalCache(refresh_prob=refresh_prob, selector=selector)
        self._cache_size = cache_size
        self.populate()

    def populate(self):
        """writeme"""
        while len(self.cache) <= self._cache_size:
            key, entity = self._selector.next()
            self.cache.add(key, entity)

    def next(self):
        """
        """
        item_buffer = []
        for n in range(self.batch_size):
            item = self.cache.next()[-1]
            for fx in self._transformers:
                item = fx(item)
            item_buffer.append(item)
        return self._serializer(item_buffer)

    def __iter__(self):
        return self
