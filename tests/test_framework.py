import biggie
import os
import pytest

import optimus
import sources


def build_model():
    x_in = optimus.Input(name="x_input", shape=(None, 2))
    class_idx = optimus.Input(name="y_target", shape=(None,), dtype='int32')
    learning_rate = optimus.Input(name='learning_rate', shape=None)

    layer0 = optimus.Affine(
        name='layer0', input_shape=x_in.shape,
        output_shape=(None, 100), act_type='tanh')

    classifier = optimus.Affine(
        name='classifier', input_shape=layer0.output.shape,
        output_shape=(None, 2), act_type='softmax')

    nll = optimus.NegativeLogLikelihoodLoss(name='nll')
    likelihoods = optimus.Output(name='likelihoods')
    loss = optimus.Output(name='loss')

    trainer_edges = optimus.ConnectionManager([
        (x_in, layer0.input),
        (layer0.output, classifier.input),
        (classifier.output, nll.likelihoods),
        (class_idx, nll.index),
        (nll.output, loss)])

    update_manager = optimus.ConnectionManager([
        (learning_rate, layer0.weights),
        (learning_rate, layer0.bias),
        (learning_rate, classifier.weights),
        (learning_rate, classifier.bias)])

    trainer = optimus.Graph(
        name='trainer',
        inputs=[x_in, class_idx, learning_rate],
        nodes=[layer0, classifier, nll],
        connections=trainer_edges.connections,
        outputs=[loss],
        loss=loss,
        updates=update_manager.connections,
        verbose=True)

    optimus.random_init(classifier.weights)

    predictor_edges = optimus.ConnectionManager([
        (x_in, layer0.input),
        (layer0.output, classifier.input),
        (classifier.output, likelihoods)])

    predictor = optimus.Graph(
        name='predictor',
        inputs=[x_in],
        nodes=[layer0, classifier],
        connections=predictor_edges.connections,
        outputs=[likelihoods])

    return trainer, predictor


def test_convergence(workspace):
    stream1 = sources.parabola((-2, 2), 2.5)
    stream2 = sources.gaussian2d((0, 5), (0.25, 0.5))

    stream = sources.batch(streams=[stream1, stream2], batch_size=50,
                           probs=[0.5, 0.5])

    trainer, predictor = build_model()
    sfile = os.path.join(workspace, 'params.hdf5')
    params = biggie.Stash(sfile)
    driver = optimus.Driver(graph=trainer, name='take000',
                            parameter_cache=params,
                            log_file='training_stats.csv')

    hyperparams = dict(learning_rate=0.02)
    stats = driver.fit(stream, hyperparams=hyperparams, save_freq=100,
                       print_freq=100, max_iter=500)

    # Verify that the output stats and checkpointed params make sense.
    num_checkpoints = 5
    assert stats.loss.iloc[0] > stats.loss.iloc[-1]
    assert len(params) == num_checkpoints
    assert stats.key.isin(params.keys()).sum() == num_checkpoints

    # Verify that saved params can be set in the predictor and used.
    for key in params.keys():
        predictor.param_values = params.get(key)
        data = next(stream)
        assert predictor(x_input=data['x_input'])
