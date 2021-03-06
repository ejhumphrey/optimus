"""Pointwise theano functions and other such malarkey.

Note: Unless these are re-written as classes, which isn't the worst idea,
a dictionary is neccesary so that the activation function can be serialized as
a string. If it were a class, it could / would have a name property.
"""

import theano.tensor as T


def linear(x):
    """Write my LaTeX form."""
    return x


def relu(x):
    """Write my LaTeX form."""
    return x * (x > 0) 


def tanh(x):
    """Write my LaTeX form."""
    return T.tanh(x)


def sigmoid(x):
    """Write my LaTeX form."""
    return T.nnet.sigmoid(x)


def softmax(x):
    return T.nnet.softmax(x)


def soft_shrink(x, threshold, Q):
    """Write my LaTeX form."""
    raise NotImplementedError("'soft_shrink' is not implemented yet.")


def hard_shrink(x, threshold):
    """Write my LaTeX form."""
    raise NotImplementedError("'hard_shrink' is not implemented yet.")


def soft_relu(x, knee=1.0):
    """Log-approximation of half-wave rectification (ReLU)

    Parameters
    ----------
    x: symbolic type
        Typically, the independent variable.
    knee: scalar (symbolic or real-valued)
        Knee of the log-approximation.

    Returns
    -------
    y: symbolic type
        Dependent variable.
    """
    return T.log(1 + T.exp(knee * x)) / knee


Activations = {'linear': linear,
               'relu': relu,
               'soft_relu': soft_relu,
               'tanh': tanh,
               'sigmoid': sigmoid,
               'soft_shrink': soft_shrink,
               'hard_shrink': hard_shrink,
               'softmax': softmax}


def l2norm(x):
    scalar = T.pow(T.pow(x, 2.0).sum(axis=1), 0.5)
    return x / scalar.dimshuffle(0, 'x')


def euclidean(a, b):
    """Row-wise euclidean distance between tensors.
    """
    a, b = a.flatten(2), b.flatten(2)
    return T.sqrt(T.sum(T.pow(a - b, 2.0), axis=1))


def manhattan(a, b):
    """Row-wise manhattan distance between tensors.
    """
    a, b = a.flatten(2), b.flatten(2)
    return T.sum(T.abs_(a - b), axis=1)


def euclidean_proj(a, b):
    """Projected Euclidean distance between tensors.
    """
    a = a.flatten(2).dimshuffle("x", 0, 1)
    b = b.flatten(2).dimshuffle(0, "x", 1)
    return T.sqrt(T.sum(T.pow(a - b, 2.0), axis=-1))


def manhattan_proj(a, b):
    """Projected Manhattan distance between tensors.
    """
    a = a.flatten(2).dimshuffle("x", 0, 1)
    b = b.flatten(2).dimshuffle(0, "x", 1)
    return T.sum(T.abs_(a - b), axis=-1)


def max_not_index(X, index):
    """Returns the maximum elements in each row of `X`, ignoring the columns
    in `index`.

    Parameters
    ----------
    X : T.matrix, ndim=2
        Input data
    index : T.ivector
        Column indexes to ignore.

    Returns
    -------
    Xmax : T.vector, dtype=X.dtype, shape=X.shape[0]
        Maximum values of each row in `X` not in columns `index`.
    """
    batch_idx = T.arange(index.shape[0], dtype='int32')
    index_values = X[batch_idx, index]
    offset = T.zeros_like(X)
    offset = T.set_subtensor(offset[batch_idx, index], 1.0)
    offset *= index_values.dimshuffle(0, 'x') + 1
    return T.max(X - offset, axis=1)


def min_not_index(X, index):
    """Returns the minimum elements in each row of `X`, ignoring the columns
    in `index`.

    Parameters
    ----------
    X : T.matrix, ndim=2
        Input data
    index : T.ivector
        Column indexes to ignore.

    Returns
    -------
    Xmin : T.vector, dtype=X.dtype, shape=X.shape[0]
        Minimum values of each row in `X` not in columns `index`.
    """
    batch_idx = T.arange(index.shape[0], dtype='int32')
    index_values = X[batch_idx, index]
    offset = T.zeros_like(X)
    offset = T.set_subtensor(offset[batch_idx, index], 1.0)
    offset *= index_values.dimshuffle(0, 'x') + 1
    return T.min(X + offset, axis=1)
