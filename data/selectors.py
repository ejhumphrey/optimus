"""writeme"""

import numpy as np


def iteritems(obj):
    """Infinite iteritem generator."""
    keys = obj.keys()
    idx = 0
    while True:
        yield keys[idx], obj.get(keys[idx])
        idx += 1
        if idx >= len(keys):
            idx = 0


def sorted_iteritems(obj):
    """Like iteritems, but keys are sorted first."""
    keys = obj.keys()
    keys.sort()
    idx = 0
    while True:
        yield keys[idx], obj.get(keys[idx])
        idx += 1
        if idx >= len(keys):
            idx = 0


def permuted_iteritems(obj):
    """Like iteritems, but a permutation of the items. Loops forever."""
    keys = obj.keys()
    order = np.arange(len(keys))
    np.random.shuffle(order)
    idx = 0
    while True:
        key = keys[order[idx]]
        yield key, obj.get(key)
        idx += 1
        if idx >= len(keys):
            idx = 0
            np.random.shuffle(order)
