"""Data stream generators.

All data streams are created via the same interface:

fx = generator(obj, keys=None, getitem=None, epochs=None, **kwargs), where
    obj: dict-like
        Object over which to iterate
    keys: list
        Keys for this object, defaults to all
    getitem: function
        Operates in the form "value = getitem(obj, key)"; by default, this
        is simply "value = obj[key]"
    epochs: int, or None
        Number of times to iterate over the object; if None, loops forever.
    kwargs: dict
        Additional arguments to pass to getitem.
"""

import numpy as np


def _getitem(obj, key):
    """
    Parameters
    ----------
    obj: dict-like
        Key-value mapping.

    """
    return obj.get(key)


def iterate(obj, keys=None, getitem=_getitem, epochs=None):
    """Infinite data stream.

    Yields
    ------
    key: str or hashable type
        Pointer to an item.
    value: object
        Item for the returned key.
    """
    if keys is None:
        keys = obj.keys()
    idx, total_epochs = 0, 0
    done = False
    while not done:
        yield keys[idx], getitem(obj, keys[idx])
        idx += 1
        if idx >= len(keys):
            idx = 0
            if not epochs is None:
                total_epochs += 1
                done = total_epochs >= epochs


def sort(obj, keys=None, getitem=_getitem, epochs=None):
    """Infinite data stream of sorted keys.

    Yields
    ------
    key: str or hashable type
        Pointer to an item.
    value: object
        Item for the returned key.
    """
    if keys is None:
        keys = obj.keys()

    keys.sort()
    return iterate(obj, keys, getitem=getitem, epochs=epochs)


def permute(obj, keys=None, getitem=_getitem, epochs=None):
    """Infinite data stream of sorted keys.

    Yields
    ------
    key: str or hashable type
        Pointer to an item.
    value: object
        Item for the returned key.
    """
    if keys is None:
        keys = obj.keys()
    order = np.arange(len(keys))
    np.random.shuffle(order)
    idx, total_epochs = 0, 0
    done = False
    while not done:
        key = keys[order[idx]]
        yield key, getitem(obj, key)
        idx += 1
        if idx >= len(keys):
            idx = 0
            np.random.shuffle(order)
            if not epochs is None:
                total_epochs += 1
                done = total_epochs >= epochs
