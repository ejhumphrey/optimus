import numpy as np
import pytest

import optimus.core as core
import optimus.nodes as nodes
import optimus.util as util
import optimus.losses as losses


def test_NegativeLogLikelihood():
    lhoods = core.Input(name='likelihoods', shape=(None, 2))
    y_true = core.Input(name='y_true', shape=(None,), dtype='int32')

    nll = losses.NegativeLogLikelihood(name='nll')

    nll.likelihoods.connect(lhoods)
    with pytest.raises(nodes.UnconnectedNodeError):
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


def test_CrossEntropyLoss():
    pred = core.Input(name='prediction', shape=(None, 2))
    target = core.Input(name='target', shape=(None, 2))

    xentropy = losses.CrossEntropyLoss(name='cross_entropy')

    xentropy.prediction.connect(pred)
    with pytest.raises(nodes.UnconnectedNodeError):
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


def test_SimilarityMargin():
    dist = core.Input(name='distance', shape=(None,))
    equiv = core.Input(name='equivalence', shape=(None,))
    sim_margin = core.Input(name='sim_margin', shape=None)
    diff_margin = core.Input(name='diff_margin', shape=None)

    contrast = losses.SimilarityMargin(name='sim_margin')

    contrast.distance.connect(dist)
    with pytest.raises(nodes.UnconnectedNodeError):
        contrast.transform()

    contrast.equivalence.connect(equiv)
    contrast.sim_margin.connect(sim_margin)
    contrast.diff_margin.connect(diff_margin)
    assert contrast.output.shape is None
    contrast.transform()

    # TODO: Fixshape
    # assert contrast.output.shape == ()
    fx = util.compile(inputs=[dist, equiv, sim_margin, diff_margin],
                      outputs=[contrast.output])

    dvals = np.array([0.5, 0.5, 2.5, 2.5])
    evals = np.array([1, 0, 1, 0])
    margins = dict(sim_margin=1.0, diff_margin=2.0)
    exps = [0, 1.5**2, 1.5**2, 0]
    for n in range(4):
        idx = slice(n, n + 1)
        cost = fx(distance=dvals[idx], equivalence=evals[idx], **margins)
        assert cost[0] == exps[n], \
            (n, dvals[idx], evals[idx], cost[0], exps[n])


def test_ContrastiveMargin_no_filter():
    cost_sim = core.Input(name='cost_sim', shape=(None,))
    cost_diff = core.Input(name='cost_diff', shape=(None,))
    margin_sim = core.Input(name='margin_sim', shape=None)
    margin_diff = core.Input(name='margin_diff', shape=None)

    contrast = losses.ContrastiveMargin(name='contrastive_margin',
                                        filter_zeros=False)

    contrast.cost_sim.connect(cost_sim)
    with pytest.raises(nodes.UnconnectedNodeError):
        contrast.transform()

    contrast.cost_diff.connect(cost_diff)
    contrast.margin_sim.connect(margin_sim)
    contrast.margin_diff.connect(margin_diff)
    assert contrast.output.shape is None
    contrast.transform()

    # TODO: Fixshape
    # assert contrast.output.shape == ()
    fx = util.compile(inputs=[cost_sim, cost_diff, margin_sim, margin_diff],
                      outputs=[contrast.output])

    c_sim = np.array([0.5, 0.1, 0.5])
    c_diff = np.array([0.5, 0, 2])
    margins = dict(margin_sim=0.25, margin_diff=2.0)
    exps = [1.5**2 + 0.25**2, 4, 0.25**2]
    for n in range(len(exps)):
        idx = slice(n, n + 1)
        cost = fx(cost_sim=c_sim[idx], cost_diff=c_diff[idx], **margins)
        assert cost[0] == exps[n]


def _relu(x):
    return x * (x > 0)


def test_ContrastiveMargin_with_filter():
    cost_sim = core.Input(name='cost_sim', shape=(None,))
    cost_diff = core.Input(name='cost_diff', shape=(None,))
    margin_sim = core.Input(name='margin_sim', shape=None)
    margin_diff = core.Input(name='margin_diff', shape=None)

    contrast = losses.ContrastiveMargin(name='contrastive_margin',
                                        filter_zeros=True)

    contrast.cost_sim.connect(cost_sim)
    with pytest.raises(nodes.UnconnectedNodeError):
        contrast.transform()

    contrast.cost_diff.connect(cost_diff)
    contrast.margin_sim.connect(margin_sim)
    contrast.margin_diff.connect(margin_diff)
    assert contrast.output.shape is None
    contrast.transform()

    # TODO: Fixshape
    # assert contrast.output.shape == ()
    fx = util.compile(inputs=[cost_sim, cost_diff, margin_sim, margin_diff],
                      outputs=[contrast.output])

    c_sim = np.array([0.5, 0.1, 0.5])
    c_diff = np.array([0.5, 0, 2])
    margins = dict(margin_sim=0.25, margin_diff=2.0)

    exp = 1.5**2 + 0.25**2
    cost = fx(cost_sim=c_sim, cost_diff=c_diff, **margins)
    assert cost[0] == exp
