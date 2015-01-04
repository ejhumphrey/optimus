"""Demonstration of building an MNIST classifier.


This script and its corresponding functionality expects the gzipped pickle file
provided courtesy of the Theano dev team at Montreal:

    http://deeplearning.net/data/mnist/mnist.pkl.gz


Sample call:
python ./examples/mnist.py \
~/mnist.pkl.gz \
./examples/sample_definition.json \
examples/sample_params.npz
"""

import argparse
import cPickle
import gzip
import numpy as np

import optimus


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
        for split in cPickle.load(fp):
            n_samples = len(split[1])
            data = np.zeros([n_samples, 1, 28, 28])
            labels = np.zeros([n_samples], dtype=int)
            for n, (x, y) in enumerate(zip(*split)):
                data[n, ...] = x.reshape(1, 28, 28)
                labels[n] = y
            dsets.append((data, labels))

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


def build_models():
    """Create a trainer and predictor graph for digit classification.

    Returns
    -------
    trainer : optimus.Graph
        Graph with in-place updates for back-propagating error.

        Parameters
        ----------
        data : np.ndarray, shape=(*, 1, 28, 28)
            Images to process.
        labels : np.ndarray, shape=(*,), dtype=int
            Target class indices of `data`.
        learning_rate : scalar, > 0
            Update parameter.

        Returns
        -------
        result : dict
            Requested outputs; keyed under 'loss' and 'likelihoods'.

    predictor : optimus.Graph
        Static graph for processing images.

        Parameters
        ----------
        data : np.ndarray, shape=(*, 1, 28, 28)
            Images to process.

        Returns
        -------
        result : dict
            Requested outputs; keyed under 'likelihoods'.
    """
    # 1.1 Create Inputs
    input_data = optimus.Input(
        name='data',
        shape=(None, 1, 28, 28))

    class_labels = optimus.Input(
        name='labels',
        shape=(None,),
        dtype='int32')

    learning_rate = optimus.Input(
        name='learning_rate',
        shape=None)

    # 1.2 Create Nodes
    conv = optimus.Conv3D(
        name='conv',
        input_shape=input_data.shape,
        weight_shape=(15, 1, 9, 9),
        pool_shape=(2, 2),
        act_type='relu')

    affine = optimus.Affine(
        name='affine',
        input_shape=conv.output.shape,
        output_shape=(None, 512,),
        act_type='relu')

    classifier = optimus.Affine(
        name='classifier',
        input_shape=affine.output.shape,
        output_shape=(None, 10),
        act_type='linear')

    softmax = optimus.Softmax(name='softmax')

    # 1.1 Create Losses
    nll = optimus.NegativeLogLikelihoodLoss(name='nll')

    # 1.2 Define outputs
    likelihoods = optimus.Output(name='likelihoods')
    loss = optimus.Output(name='loss')

    # 2. Define Edges
    base_edges = [
        (input_data, conv.input),
        (conv.output, affine.input),
        (affine.output, classifier.input),
        (classifier.output, softmax.input),
        (softmax.output, likelihoods)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (softmax.output, nll.likelihoods),
            (class_labels, nll.index),
            (nll.output, loss)])

    update_manager = optimus.ConnectionManager([
        (learning_rate, conv.weights),
        (learning_rate, conv.bias),
        (learning_rate, affine.weights),
        (learning_rate, affine.bias),
        (learning_rate, classifier.weights),
        (learning_rate, classifier.bias)])

    trainer = optimus.Graph(
        name='mnist_trainer',
        inputs=[input_data, class_labels, learning_rate],
        nodes=[conv, affine, classifier, softmax, nll],
        connections=trainer_edges.connections,
        outputs=[loss, likelihoods],
        loss=loss,
        updates=update_manager.connections,
        verbose=True)

    for node in conv, affine, classifier:
        optimus.random_init(node.weights, mean=0.0, std=0.1)

    predictor_edges = optimus.ConnectionManager(base_edges)

    predictor = optimus.Graph(
        name='mnist_classifier',
        inputs=[input_data],
        nodes=[conv, affine, classifier, softmax],
        connections=predictor_edges.connections,
        outputs=[likelihoods],
        verbose=True)

    return trainer, predictor


def main(args):
    # Create the models, and pass the trainer to a driver.
    trainer, predictor = build_models()
    driver = optimus.Driver(graph=trainer)

    # Load data and configure the minibatch generator.
    train, valid, test = load_mnist(args.mnist_file)
    stream = minibatch(train[0], train[1], 50)
    hyperparams = dict(learning_rate=0.02)

    # And we're off!
    driver.fit(stream, hyperparams=hyperparams, max_iter=500, print_freq=20)

    # Note that, because they were made at the same time, `trainer` and
    #   `predictor` share the same parameters.
    y_pred = predictor(valid[0])['likelihoods'].argmax(axis=1)
    print "Accuracy: {0:0.4}".format(np.equal(valid[1], y_pred).mean())

    # Now that it's done, save the graph and parameters to disk.
    optimus.save(predictor, args.def_file, args.param_file)

    # And, because we can, let's recreate it purely from disk.
    predictor2 = optimus.load(args.def_file, args.param_file)
    y_pred2 = predictor2(valid[0])['likelihoods'].argmax(axis=1)
    print "Consistency: {0:0.4}".format(np.equal(y_pred, y_pred2).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("mnist_file",
                        metavar="mnist_file", type=str,
                        help="Path to the MNIST data file.")
    # Outputs
    parser.add_argument("def_file",
                        metavar="def_file", type=str,
                        help="Path to save the classifier graph.")
    parser.add_argument("param_file",
                        metavar="param_file", type=str,
                        help="Path to save the classifier parameters.")
    main(parser.parse_args())
