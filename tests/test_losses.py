import numpy as np
import pytest

import optimus.core as core
import optimus.util as util
import optimus.losses as losses


def test_NegativeLogLikelihood():
    lhoods = core.Input(name='likelihoods', shape=(None, 2))
    y_true = core.Input(name='y_true', shape=(None,), dtype='int32')

    nll = losses.NegativeLogLikelihood(name='nll')

    nll.likelihoods.connect(lhoods)
    with pytest.raises(AssertionError):
        nll.transform()

    nll.index.connect(y_true)
    assert nll.output.shape is None
    nll.transform()

    # TODO: Fixshape
    # assert nll.output.shape == ()
    fx = util.compile(inputs=[lhoods, y_true], outputs=[nll.output])

    x_obs = np.array([[0.001, 1 - 0.001], [0, 1]])
    y_obs = np.array([0, 1])
    # Pretty wrong answer
    assert fx(likelihoods=x_obs[:1], y_true=y_obs[:1])[0] > 6.0
    # Right answer
    assert fx(likelihoods=x_obs[1:], y_true=y_obs[1:])[0] == 0


def test_CrossEntropy():
    pred = core.Input(name='prediction', shape=(None, 2))
    target = core.Input(name='target', shape=(None, 2))

    xentropy = losses.CrossEntropy(name='cross_entropy')

    xentropy.prediction.connect(pred)
    with pytest.raises(AssertionError):
        xentropy.transform()

    xentropy.target.connect(target)
    assert xentropy.output.shape is None
    xentropy.transform()

    # TODO: Fixshape
    # assert xentropy.output.shape == ()
    fx = util.compile(inputs=[pred, target], outputs=[xentropy.output])

    x_obs = np.array([[0.001, 1 - 0.001], [0.001, 1 - 0.001]])
    y_obs = np.array([[1, 0], [0, 1]])
    # Pretty wrong answer
    assert fx(prediction=x_obs[:1], target=y_obs[:1])[0] > 6
    # Right answer
    assert fx(prediction=x_obs[1:], target=y_obs[1:])[0] < 0.01
