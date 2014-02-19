"""writeme"""

import numpy as np


def ordered_items(obj):
    """writeme"""
    for key in obj.keys():
        yield key, obj.get(key)


def sorted_items(obj):
    """writeme"""
    keys = obj.keys()
    keys.sort()
    for key in keys:
        yield key, obj.get(key)


def permute_items(obj):
    """Infinite permutation"""
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
