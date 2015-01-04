"""Demonstration of building an MNIST classifier.
"""

import cPickle
import gzip

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
    train, valid, test:
        Digits are keyed by string integers in the order they were added.
    """
    with gzip.open(gzfile, 'rb') as fp:
        train, valid, test = cPickle.load(fp)

    return train, valid, test


def build_model():
    # 1.1 Create Inputs
    input_data = optimus.Input(
        name='image',
        shape=(None, 1, 28, 28))

    class_labels = optimus.Input(
        name='label',
        shape=(None,),
        dtype='int32')

    decay = optimus.Input(
        name='decay_param',
        shape=None)

    sparsity = optimus.Input(
        name='sparsity_param',
        shape=None)

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
        n_out=10,
        act_type='linear')

    softmax = optimus.Softmax(name='softmax')

    # 1.1 Create Losses
    nll = optimus.NegativeLogLikelihoodLoss(name='nll')
    conv_decay = optimus.L2Magnitude(name='weight_decay')
    affine_sparsity = optimus.L1Magnitude(name="feature_sparsity")
    loss_sum = optimus.Add(name='loss_sum', num_inputs=3)

    loss_nodes = [nll, conv_decay, affine_sparsity]

    # 1.2 Define outputs
    likelihoods = optimus.Output(name='likelihoods')
    total_loss = optimus.Output(name='total_loss')

    # 2. Define Edges
    base_edges = [
        (input_data, conv.input),
        (conv.output, affine.input),
        (affine.output, classifier.input),
        (classifier.output, softmax.input),
        (softmax.output, likelihoods)
    ]

    trainer_edges = optimus.ConnectionManager(
        base_edges + [
            (softmax.output, nll.likelihoods),
            (class_labels, nll.index),
            (conv.weights, conv_decay.input),
            (decay, conv_decay.weight),
            (affine.output, affine_sparsity.input),
            (sparsity, affine_sparsity.weight),
            (nll.output, loss_sum.input_0),
            (conv_decay.output, loss_sum.input_1),
            (affine_sparsity.output, loss_sum.input_2),
            (loss_sum.output, total_loss)])

    update_manager = optimus.ConnectionManager([
        (learning_rate, conv.weights),
        (learning_rate, conv.bias),
        (learning_rate, affine.weights),
        (learning_rate, affine.bias),
        (learning_rate, classifier.weights),
        (learning_rate, classifier.bias)])

    trainer = optimus.Graph(
        name='mnist_3layer',
        inputs=[input_data, class_labels, decay, sparsity, learning_rate],
        nodes=[conv, affine, classifier, softmax, nll,
               conv_decay, affine_sparsity, loss_sum],
        connections=trainer_edges.connections,
        outputs=[total_loss, likelihoods],
        loss=total_loss,
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
        outputs=[likelihoods])

    return trainer, predictor


def fit():
    # 3. Create Data
    datasets = load_mnist("/Users/ejhumphrey/Desktop/mnist.pkl")
    source = optimus.Queue(
        datasets[0], batch_size=50, refresh_prob=0.0, cache_size=50000)

    driver = optimus.Driver(graph=trainer, name='example_classifier')

    hyperparams = {
        learning_rate.name: 0.02,
        sparsity.name: 0.0,
        decay.name: 0.0}

    driver.fit(source, hyperparams=hyperparams, max_iter=5000, print_freq=25)
