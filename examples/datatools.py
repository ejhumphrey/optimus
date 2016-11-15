import gzip
import numpy as np
import pickle


def load_mnist(mnist_file):
    """Load the MNIST dataset into memory.

    Parameters
    ----------
    mnist_file : str
        Path to gzipped MNIST file.

    Returns
    -------
    train, valid, test: tuples of np.ndarrays
        Each consists of (data, labels), where data.shape=(N, 1, 28, 28) and
        labels.shape=(N,).
    """
    dsets = []
    with gzip.open(mnist_file, 'rb') as fp:
        for split in pickle.load(fp):
            n_samples = len(split[1])
            data = np.zeros([n_samples, 1, 28, 28])
            labels = np.zeros([n_samples], dtype=int)
            for n, (x, y) in enumerate(zip(*split)):
                data[n, ...] = x.reshape(1, 28, 28)
                labels[n] = y
            dsets.append((data, labels))

    return dsets


def load_mnist_npz(mnist_file):
    """Load the MNIST dataset into memory from an NPZ.

    Parameters
    ----------
    mnist_file : str
        Path to an NPZ file of MNIST data.

    Returns
    -------
    train, valid, test: tuples of np.ndarrays
        Each consists of (data, labels), where data.shape=(N, 1, 28, 28) and
        labels.shape=(N,).
    """
    data = np.load(mnist_file)
    dsets = []
    for name in 'train', 'valid', 'test':
        x = data['x_{}'.format(name)].reshape(-1, 1, 28, 28)
        y = data['y_{}'.format(name)]
        dsets.append([x, y])

    return dsets


def minibatch(data, labels, batch_size, max_iter=np.inf):
    """Random mini-batches generator.

    Parameters
    ----------
    data : array_like, len=N
        Observation data.
    labels : array_like, len=N
        Labels corresponding the the given data.
    batch_size : int
        Number of datapoints to return at each iteration.
    max_iter : int, default=inf
        Number of iterations before raising a StopIteration.

    Yields
    ------
    batch : dict
        Random batch of datapoints, under the keys `data` and `labels`.
    """
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same number of items.")

    num_points = len(labels)
    if num_points <= batch_size:
        raise ValueError("batch_size cannot exceed number of data points")

    count = 0
    order = np.random.permutation(num_points)
    idx = 0
    while count < max_iter:
        x, y = [], []
        while len(y) < batch_size:
            x.append(data[order[idx]])
            y.append(labels[order[idx]])
            idx += 1
            if idx >= num_points:
                idx = 0
                np.random.shuffle(order)
        yield dict(data=np.asarray(x), labels=np.asarray(y))
        count += 1
