"""writeme"""

import numpy as np


def iteritems(source):
    """Infinite iteritem generator."""
    keys = source.keys()
    idx = 0
    while True:
        yield keys[idx], source.get(keys[idx])
        idx += 1
        if idx >= len(keys):
            idx = 0


def sorted_iteritems(source):
    """Like iteritems, but keys are sorted first."""
    keys = source.keys()
    keys.sort()
    idx = 0
    while True:
        yield keys[idx], source.get(keys[idx])
        idx += 1
        if idx >= len(keys):
            idx = 0


def permuted_iteritems(source):
    """Like iteritems, but a permutation. Loops forever."""
    keys = source.keys()
    order = np.arange(len(keys))
    np.random.shuffle(order)
    idx = 0
    while True:
        key = keys[order[idx]]
        yield key, source.get(key)
        idx += 1
        if idx >= len(keys):
            idx = 0
            np.random.shuffle(order)
