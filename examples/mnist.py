"""Demonstration of building an MNIST classifier.
"""

import cPickle
import gzip
import numpy as np

import optimus


def load_mnist(gzfile):
    """Load the MNIST dataset into memory.

    Expects the pre-processed data provided courtesy of the Theano devs:
        http://deeplearning.net/data/mnist/mnist.pkl.gz

    Parameters
    ----------
    gzfile : str
        Path to gzipped MNIST file.

    Returns
    -------
    train, valid, test: tuples of np.ndarrays
        Each consists of (data, labels), where data.shape=(N, 1, 28, 28) and
        labels.shape=(N,).
    """
    dsets = []
    with gzip.open(gzfile, 'rb') as fp:
        for split in cPickle.load(fp):
            n_samples = len(split[1])
            data = np.zeros([n_samples, 1, 28, 28])
            labels = np.zeros([n_samples], dtype=int)
            for n, (x, y) in enumerate(zip(*split)):
                data[n, ...] = x.reshape(1, 28, 28)
                labels[n] = y
            dsets.append((data, labels))

    return dsets


def build_model():
    # 1.1 Create Inputs
    input_data = optimus.Input(
        name='image',
        shape=(None, 1, 28, 28))

    class_labels = optimus.Input(
        name='label',
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


def main(mnist_file):
    # 3. Create Data
    train, valid, test = load_mnist(mnist_file)
    source = optimus.Queue(
        train, batch_size=50, refresh_prob=0.0, cache_size=50000)

    trainer, predictor = build_model()

    driver = optimus.Driver(graph=trainer, name='example_classifier')

    hyperparams = dict(learning_rate=0.02, sparsity_param=0.0,
                       decay_param=0.0)

    driver.fit(source, hyperparams=hyperparams, max_iter=5000, print_freq=25)
