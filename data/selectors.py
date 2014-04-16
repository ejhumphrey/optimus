"""writeme"""

import numpy as np


def iteritems(obj):
    """Iteritem generator."""
    for key in obj.keys():
        yield key, obj.get(key)


def sorted_iteritems(obj):
    """Like iteritems, but keys are sorted first."""
    keys = obj.keys()
    keys.sort()
    for key in keys:
        yield key, obj.get(key)


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
