import numpy as np
import theano


def compile(inputs, outputs):
    """Thin wrapper around theano's function compilation.

    Parameters
    ----------
    inputs : list of optimus.Inputs
        Optimus inputs.

    outputs : list of optimus.Outputs
        Connected and transformed outputs.

    Returns
    -------
    func : callable
        Function defined by the graph of inputs / outputs.
    """
    return theano.function(inputs=[x.variable for x in inputs],
                           outputs=[z.variable for z in outputs],
                           allow_input_downcast=True)


def random_init(param, mean=0.0, std=0.025):
    """Initialize a parameter from a normal distribution.

    Parameters
    ----------
    param : optimus.core.Parameter
        Object to initialize.
    mean : scalar
        Average of the distribution.
    std : scalar
        Standard deviation of the distribution.
    """
    param.value = np.random.normal(mean, std, size=param.shape)


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
        pad = np.zeros([length / 2] + list(value.shape[1:]))

    if not pad is None:
        pad[:] = fill_value
        value = np.concatenate([pad, value, pad[:-1]], axis=0)

    idx = 0
    sample = value[idx:idx + length]
    while sample.shape[0] == length:
        yield sample.transpose(axes_reorder)
        idx += stride
        sample = value[idx:idx + length]


def concatenate_data(batches):
    """Concatenate a set of batches.

    Parameters
    ----------
    batches: list
        A set of batches to combine.

    Returns
    -------
    batch: dict
        Combination of the input batches.
    """
    result = dict()
    for batch in batches:
        for key in batch:
            if not key in result:
                result[key] = list()
            result[key].append(batch[key])
    for key in result:
        result[key] = np.concatenate(result[key], axis=0)
    return result
