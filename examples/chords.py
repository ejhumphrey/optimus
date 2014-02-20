"""writeme"""

from .. import data
from random import choice
import numpy as np


class CQTEntity(data.Entity):
    """writeme"""
    def __init__(self, cqt, label):
        """writeme"""
        data.Entity.__init__(
            self, cqt=data.Feature(cqt), label=data.Feature(label))


class CQTFactory(data.Factory):
    """writeme"""
    def __init__(self, source, batch_size, transformers=None, selector=None,
                 cache_size=1000, refresh_prob=0.01):
        """writeme"""
        self._values = []
        self._labels = []

        data.Factory.__init__(
            self,
            source=source,
            batch_size=batch_size,
            transformers=transformers,
            selector=selector,
            cache_size=cache_size,
            refresh_prob=refresh_prob)

    def values(self):
        """writeme"""
        return np.asarray(self._values)

    def labels(self):
        """writeme"""
        return np.asarray(self._labels)

    def buffer(self, entity):
        """writeme"""
        self._values.append(entity.cqt.value)
        self._labels.append(entity.label.value)

    @property
    def ready(self):
        """writeme"""
        return len(self.values()) >= self.batch_size

    def clear(self):
        """writeme"""
        self._values = []
        self._labels = []


class CQTSampler(object):
    """writeme"""
    def __init__(self, left, right, fill_value):
        self.left = left
        self.right = right
        self.fill_value = fill_value

    def __call__(self, entity):
        """writeme"""
        index = np.random.randint(low=0, high=len(entity.cqt.value))
        cqt = context_slice(
            entity.cqt.value, index, self.left, self.right, self.fill_value)
        label = entity.label.value[index]
        return CQTEntity(cqt, label)


def context_slice(value, index, left, right, fill_value):
    """Slice a sequence with context, padding values out of range.

    Parameters
    ----------
    value : np.ndarray
        Multidimensianal array to slice.
    index : int
        Position along the first axis to center the slice.
    left : int
        Number of previous points to return.
    right : int
        Number of subsequent points to return.
    fill_value : scalar
        Value to use for out-of-range regions.

    Returns
    -------
    region : np.ndarray
        Slice of length left + right + 1; all other dims are equal to the
        input.
    """
    idx_left = max([index - left, 0])
    idx_right = min([index + right + 1, len(value)])
    observation = value[idx_left:idx_right]
    if isinstance(value, np.ndarray):
        other_dims = list(value.shape[1:])
        result = np.empty([left + right + 1] + other_dims,
                          dtype=value.dtype)
        result[:] = fill_value
    else:
        result = [fill_value] * (left + right + 1)
    idx_out = idx_left - (index - left)
    result[idx_out:idx_out + len(observation)] = observation
    return result
