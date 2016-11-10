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
    stride: int, default=1
        Number of data points to advance at each step.
    axis: int, default=0
        Axis to step through.
    mode: str, default='full'
        Like convolution, one of ['valid', 'full', 'same']
    fill_value: scalar, default=0.0
        Like convolution, value to use for out-of-bounds values.

    Yields
    ------
    value_slice: np.ndarray
        Slices of value, with the 'axis' dimension of size 'length'.
    """
    axes_order = list(range(value.ndim))
    axes_order.insert(0, axes_order.pop(axis))
    axes_reorder = np.array(axes_order).argsort()
    value = value.transpose(axes_order)
    pad = None
    if mode == 'full':
        pad = np.zeros([length] + list(value.shape[1:]))
    elif mode == 'same':
        pad = np.zeros([int(length / 2)] + list(value.shape[1:]))

    if pad is not None:
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
            if key not in result:
                result[key] = list()
            result[key].append(batch[key])
    for key in result:
        result[key] = np.concatenate(result[key], axis=0)
    return result


def convolve(x_in, graph, axis=0, chunk_size=250):
    """Apply a graph convolutionally to a field in an an entity.

    Parameters
    ----------
    x_in : np.ndarray
        Observation to predict.
    graph : optimus.Graph
        Network for processing an entity.
    data_key : str
        Name of the field to use for the input.
    chunk_size : int, default=None
        Number of slices to transform in a given step. When None, parses one
        slice at a time.

    Returns
    -------
    z_out : dict
        Output values mapped under their respective keys.
    """
    # TODO(ejhumphrey): Make this more stable, somewhat fragile as-is
    win_length = graph.inputs.values()[0].shape[2]
    input_stepper = array_stepper(
        x_in, win_length, axis=axis, mode='same')
    results = dict([(k, list()) for k in graph.outputs])
    if chunk_size:
        chunk = []
        for x in input_stepper:
            chunk.append(x)
            if len(chunk) == chunk_size:
                for k, v in graph(np.array(chunk)).items():
                    results[k].append(v)
                chunk = []
        if len(chunk):
            for k, v in graph(np.array(chunk)).items():
                results[k].append(v)
    else:
        for x in input_stepper:
            for k, v in graph(x[np.newaxis, ...]).items():
                results[k].append(v)
    for k in results:
        results[k] = np.concatenate(results[k], axis=0)

    return results
