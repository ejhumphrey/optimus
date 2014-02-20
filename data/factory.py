"""writeme"""


from . import sources
from . import selectors


class Factory(object):
    """writeme"""
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

        self.cache = sources.LocalCache(
            refresh_prob=refresh_prob, selector=selector)
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
