import numpy as np


def array_stepper(value, length, stride=1, axis=0,
                  mode='full', fill_value=0.0):
    """Generator yielding slices of an input array along a given axis.

    Parameters
    ----------
    value: np.ndarray
        Array to step through.
    length: int
        Length of the sub-slice.
    stride: int
        Number of data points to advance at each step.
    axis: int
        Axis to step through.
    mode: str
        Like convolution, one of ['valid', 'full', 'same']
    fill_value: scalar
        Like convolution, value to use for out-of-bounds values.

    Yields
    ------
    value_slice: np.ndarray
        Slices of value, with the 'axis' dimension of size 'length'.
    """
    axes_order = range(value.ndim)
    axes_order.insert(0, axes_order.pop(axis))
    axes_reorder = np.array(axes_order).argsort()
    value = value.transpose(axes_order)
    pad = None
    if mode == 'full':
        pad = np.zeros([length] + list(value.shape[1:]))
    elif mode == 'same':
        pad = np.zeros([length/2] + list(value.shape[1:]))

    if not pad is None:
        pad[:] = fill_value
        value = np.concatenate([pad, value, pad], axis=0)

    idx = 0
    sample = value[idx:idx + length]
    while sample.shape[0] == length:
        yield sample.transpose(axes_reorder)
        idx += stride
        sample = value[idx:idx + length]
