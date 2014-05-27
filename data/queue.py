"""writeme"""


from . import selectors
import numpy as np
import sys


class Cache(object):
    """Basically, a dictionary with some bonus methods."""
    def __init__(self, refresh_prob, selector=None):
        """writeme."""

        if selector is None:
            selector = selectors.permuted_iteritems
        self._selector = selector

        self._data = dict()
        self._refresh_prob = refresh_prob
        self._item_gen = self._selector(self._data)

    def __update_selector__(self):
        self._item_gen = self._selector(self._data)

    def get(self, key):
        """Fetch the value for a given key."""
        return self._data.get(key)

    def add(self, key, entity):
        """writeme."""
        self._data[key] = entity
        self.__update_selector__()

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
            del self._data[key]
            self.__update_selector__()

    def next(self):
        """Return the next item from the internal selector."""
        key, entity = self._item_gen.next()
        self.remove(key, prob=self._refresh_prob)
        return key, entity


def unpack_entities(entities):
    """Turn a set of entities into key-np.ndarray objects.

    TODO(ejhumphrey): Speed this method up. It is a bottleneck in the data
    bundling / presentation process.

    Parameters
    ----------
    entities: list of Entities
        Note that all Entities must have identical fields.

    Returns
    -------
    arrays: dict of np.ndarrays
        Values in 'arrays' are keyed by the fields of the Entities.
    """
    data = dict([(k, list()) for k in entities[0].keys()])
    for entity in entities:
        for k, v in entity.values.iteritems():
            data[k].append(v)

    for k in data:
        data[k] = np.asarray(data[k])
    return data


class Queue(object):
    """
    There are two levels of data selection:
    - From the source, default=permutation
    - From the local cache, default=



    Data Pipeline:

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
    def __init__(self, source, batch_size, cache_size=1000, refresh_prob=0.01,
                 transformers=None, selector=None, serializer=None):
        """writeme"""
        self._source = source
        self.batch_size = batch_size

        # If None, safely create list
        if transformers is None:
            transformers = list()
        self._transformers = transformers

        # If None, create default selector
        if selector is None:
            selector = selectors.permuted_iteritems
        self._selector = selector(self._source)

        if serializer is None:
            serializer = unpack_entities
        self._serializer = serializer

        # If the requested cache size is larger than the number of items in
        # the source, set the capacity to the size of the source and disable
        # auto-refresh.
        if cache_size > len(source) or cache_size is None:
            cache_size = len(source)
            refresh_prob = 0

        self.cache = Cache(refresh_prob=refresh_prob, selector=selector)
        self._cache_size = cache_size
        self.populate(cache_size > 1)

    def populate(self, verbose=False):
        """Cache datapoints in memory."""
        if verbose:
            count = self._cache_size - len(self.cache)
            sys.stdout.write("Loading %d datapoint(s) " % count)
            sys.stdout.flush()
        dots = 0
        while len(self.cache) < self._cache_size:
            key, entity = self._selector.next()
            self.cache.add(key, entity)
            if verbose and (10 * len(self.cache) / float(count)) > dots:
                dots += 1
                sys.stdout.write(".")
                sys.stdout.flush()
        if verbose:
            sys.stdout.write(" Done!\n")
            sys.stdout.flush()

    def next(self):
        """Return the next batch of datapoints.

        Any datapoints in the pipeline that return 'None' are discarded.
        """
        data_buffer = dict()
        batch_idx = 0
        while batch_idx < self.batch_size:
            key, entity = self.cache.next()
            # Apply the Transformer pipeline
            for fx in self._transformers:
                entity = fx(entity)
                if entity is None:
                    break
            # Drop null datapoints
            if entity is None:
                continue
            for field, value in entity.values.iteritems():
                if not field in data_buffer:
                    val_shape = [self.batch_size] + list(value.shape)
                    data_buffer[field] = np.zeros(val_shape, dtype=value.dtype)
                data_buffer[field][batch_idx] = value
            batch_idx += 1
        self.populate()
        return data_buffer

    def __iter__(self):
        return self
