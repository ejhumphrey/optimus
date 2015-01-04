"""writeme."""

import theano.tensor as T
from . import core
from . import functions as FX


class Loss(core.JObject):

    def __init__(self, name, **kwargs):
        """writeme."""
        self.name = name
        self.__args__ = dict(**kwargs)
        self.loss = core.Port(
            shape=[], name=self.__own__("loss"))
        self.cost = core.Port(
            shape=None, name=self.__own__("cost"))

    def is_ready(self):
        """Return true when all input ports are loaded."""
        set_inputs = all([p.variable for p in self.inputs.values()])
        set_outputs = all([p.variable for p in self.outputs.values()])
        return set_inputs and not set_outputs

    def reset(self):
        """writeme"""
        for p in self.inputs.values():
            p.reset()
        for p in self.outputs.values():
            p.reset()

    @property
    def __json__(self):
        self.__args__.update(type=self.type, name=self.name)
        return self.__args__

    def __repr__(self):
        """Render the object as an unambiguous string."""
        return '<%s: %s>' % (self.type, self.name)

    def __own__(self, name):
        return "%s.%s" % (self.name, name)

    @property
    def params(self):
        return {}

    # --- Subclassed methods ---
    def transform(self):
        """writeme"""
        raise NotImplementedError("Subclass me!")

    @property
    def inputs(self):
        """Return a list of all active Inputs in the node."""
        raise NotImplementedError("Subclass me")

    @property
    def outputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        return dict([(v.name, v) for v in [self.loss, self.cost]])


class NegativeLogLikelihood(Loss):
    """writeme"""
    def __init__(self, name, weighted=False):
        # Input Validation
        Loss.__init__(self, name=name, weighted=weighted)
        self.likelihood = core.Port(
            name=self.__own__("likelihood"))
        self.target_idx = core.Port(
            name=self.__own__("target_idx"), shape=[])
        self.weight = False if not weighted else core.Port(
            name=self.__own__("weight"))
        self._weighted = weighted

    @property
    def inputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        ports = [self.likelihood, self.target_idx]
        if self._weighted:
            ports.append(self.weight)
        return dict([(v.name, v) for v in ports])

    def transform(self):
        """writeme"""
        assert self.is_ready()
        assert self.likelihood.variable, "Port error: 'likelihood' not set."
        likelihood = self.likelihood.variable
        assert self.target_idx.variable, "Port error: 'target_idx' not set."
        target_idx = self.target_idx.variable

        # Create the local givens.
        batch_idx = T.arange(target_idx.shape[0], dtype='int32')
        self.loss.variable = -T.log(likelihood)[batch_idx, target_idx]
        if self._weighted:
            assert self.weight.variable, "Port error: 'weight' not set."
            self.loss.variable *= self.weight.variable
        self.cost.variable = T.mean(self.loss.variable)


class MultiNegativeLogLikelihood(NegativeLogLikelihood):
    """writeme"""
    def __init__(self, name, n_dim):
        # Input Validation
        NegativeLogLikelihood.__init__(self, name=name)
        self.__args__.update(n_dim=n_dim)
        self.n_dim = n_dim

    def transform(self):
        """writeme"""
        assert self.likelihood.variable, "Port error: 'likelihood' not set."
        likelihood = self.likelihood.variable
        assert self.target_idx.variable, "Port error: 'target_idx' not set."
        target_idx = self.target_idx.variable
        # Create the local givens.
        batch_idx = T.arange(target_idx.shape[0], dtype='int32')
        losses = []
        for i in xrange(self.n_dim):
            loss_idx = likelihood[batch_idx, i, target_idx[batch_idx, i]]
            losses.append(loss_idx.dimshuffle(0, 'x'))

        self.loss.variable = -T.log(T.concatenate(losses, axis=1))
        self.cost.variable = T.mean(self.loss.variable)


class ConditionalNegativeLogLikelihood(Loss):
    """writeme"""
    def __init__(self, name, weighted=False):
        # Input Validation
        Loss.__init__(self, name=name, weighted=weighted)
        self.likelihood = core.Port(
            name=self.__own__("likelihood"))
        self.independent_idx = core.Port(
            name=self.__own__("independent_idx"), shape=[])
        self.conditional_idx = core.Port(
            name=self.__own__("conditional_idx"), shape=[])
        self.weight = False if not weighted else core.Port(
            name=self.__own__("weight"))
        self._weighted = weighted

    @property
    def inputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        ports = [self.likelihood, self.independent_idx, self.conditional_idx]
        if self._weighted:
            ports.append(self.weight)
        return dict([(v.name, v) for v in ports])

    def transform(self):
        """writeme"""
        assert self.likelihood.variable, "Port error: 'likelihood' not set."
        likelihood = self.likelihood.variable
        assert self.independent_idx.variable, \
            "Port error: 'independent_idx' not set."
        independent_idx = self.independent_idx.variable
        assert self.conditional_idx.variable, \
            "Port error: 'conditional_idx' not set."
        conditional_idx = self.conditional_idx.variable

        # Create the local givens.
        batch_idx = T.arange(independent_idx.shape[0], dtype='int32')
        self.loss.variable = -T.log(likelihood)[batch_idx,
                                                independent_idx,
                                                conditional_idx]
        if self._weighted:
            assert self.weight.variable, "Port error: 'weight' not set."
            self.loss.variable *= self.weight.variable
        self.cost.variable = T.mean(self.loss.variable)


class Margin(Loss):
    """

    """
    def __init__(self, name, weighted=False, mode='max'):
        # Input Validation
        Loss.__init__(self, name=name, weighted=weighted, mode=mode)
        self.prediction = core.Port(
            name=self.__own__("prediction"))
        self.target_idx = core.Port(
            name=self.__own__("target_idx"), shape=[])
        self.margin = core.Port(
            name=self.__own__("margin"))
        self.weight = False if not weighted else core.Port(
            name=self.__own__("weight"))
        self.mode = mode

    @property
    def inputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        ports = [self.prediction, self.target_idx, self.margin]
        if self.weight:
            ports.append(self.weight)
        return dict([(v.name, v) for v in ports])

    def transform(self):
        """writeme"""
        assert self.prediction.variable, "Port error: 'prediction' not set."
        prediction = self.prediction.variable
        assert self.target_idx.variable, "Port error: 'target_idx' not set."
        target_idx = self.target_idx.variable
        assert self.margin.variable, "Port error: 'margin' not set."
        margin = self.margin.variable

        batch_idx = T.arange(target_idx.shape[0], dtype='int32')
        target_values = prediction[batch_idx, target_idx]

        if self.mode == 'max':
            moia_values = FX.max_not_index(prediction, target_idx)
            difference = margin + moia_values - target_values
        elif self.mode == 'min':
            moia_values = FX.min_not_index(prediction, target_idx)
            difference = margin + target_values - moia_values
        else:
            raise ValueError(
                "`mode` must take one of ['min', 'max']" % self.mode)

        self.loss.variable = FX.relu(difference)
        if self.weight:
            assert self.weight.variable, "Port error: 'weight' not set."
            self.loss.variable *= self.weight.variable
        self.cost.variable = T.mean(self.loss.variable)


class ClassificationError(Loss):
    """writeme"""
    def __init__(self, name, weighted=False, mode='max'):
        # Input Validation
        Loss.__init__(self, name=name, weighted=weighted, mode=mode)
        self.prediction = core.Port(
            name=self.__own__("prediction"))
        self.target_idx = core.Port(
            name=self.__own__("target_idx"), shape=[])
        self.weights = core.Port(name=self.__own__("weights"),
                                 shape=[]) if weighted else False
        assert mode in ['min', 'max']
        self.mode = mode

    @property
    def inputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        ports = [self.prediction, self.target_idx]
        if self.weights:
            ports.append(self.weights)
        return dict([(v.name, v) for v in ports])

    def transform(self):
        """Compute the outputs for this loss."""
        assert self.prediction.variable, "Port error: 'prediction' not set."
        prediction = self.prediction.variable

        assert self.target_idx.variable, "Port error: 'target_idx' not set."
        target_idx = self.target_idx.variable

        batch_idx = T.arange(target_idx.shape[0], dtype='int32')
        target_values = prediction[batch_idx, target_idx]

        if self.mode == 'max':
            moia_values = FX.max_not_index(prediction, target_idx)
            difference = moia_values - target_values
        elif self.mode == 'min':
            moia_values = FX.min_not_index(prediction, target_idx)
            difference = target_values - moia_values
        else:
            raise ValueError(
                "`mode` must take one of ['min', 'max']" % self.mode)

        self.loss.variable = FX.sigmoid(difference)
        if self.weights:
            self.loss.variable *= self.weights.variable
        self.cost.variable = T.mean(self.loss.variable)


class ContrastiveDivergence(Loss):
    _DISTANCE = 'distance'
    _SCORE = 'score'
    _MARGIN = 'margin'

    def __init__(self, distance, score, margin):
        raise NotImplementedError("come back to this")
        # Input Validation
        self.update(distance=distance, score=score, margin=margin)
        Loss.__init__(self)

    def loss(self, variables):
        """
        variables : dict
            Set of URL-keyed variables from which to select.
        """
        distance = variables[self[self._DISTANCE]]
        # Create the local givens.
        score = T.ivector(name=self[self._SCORE])
        margin = T.scalar(name=self[self._MARGIN])
        diff_loss = 0.5 * (1.0 - score) * self._diff_cost(distance, margin)
        same_loss = 0.5 * score * self._same_cost(distance)
        scalar_loss = T.mean(diff_loss + same_loss)
        return scalar_loss, {score.name: score, margin.name: margin}

    def _same_cost(self, x):
        return T.pow(x, 2.0)

    def _diff_cost(self, x, margin):
        return T.pow(FX.soft_hinge(margin, x), 2.0)


class L1Magnitude(Loss):
    def __init__(self, name):
        Loss.__init__(self, name=name)
        self.input = core.Port(
            name=self.__own__("input"))
        self.weight = core.Port(
            name=self.__own__("weight"))

    @property
    def inputs(self):
        """Return a dict of all active Outputs in the node."""
        # Filter based on what is set / active?
        return dict([(v.name, v) for v in [self.input, self.weight]])

    @property
    def outputs(self):
        """Return a dict of all active Outputs in the node."""
        # Filter based on what is set / active?
        return dict([(v.name, v) for v in [self.cost]])

    def transform(self):
        """writeme"""
        assert self.input.variable, "Port error: '%s' not set." % self.input
        assert self.weight.variable, "Port error: '%s' not set." % self.weight
        var_magnitude = T.sum(T.abs_(self.input.variable))
        self.cost.variable = var_magnitude * self.weight.variable


class L2Magnitude(L1Magnitude):

    def transform(self):
        """writeme"""
        assert self.input.variable, "Port error: '%s' not set." % self.input
        assert self.weight.variable, "Port error: '%s' not set." % self.weight
        var_magnitude = T.pow(T.sum(T.pow(self.input.variable, 2.0)), 0.5)
        self.cost.variable = var_magnitude * self.weight.variable


class Max(Loss):
    def __init__(self, name):
        Loss.__init__(self, name=name)
        self.input = core.Port(
            name=self.__own__("input"))
        self.weight = core.Port(
            name=self.__own__("weight"))
        self.threshold = core.Port(
            name=self.__own__("threshold"))

    @property
    def inputs(self):
        """Return a dict of all active Outputs in the node."""
        # Filter based on what is set / active?
        return dict([(v.name, v)
                     for v in [self.input, self.weight, self.threshold]])

    @property
    def outputs(self):
        """Return a dict of all active Outputs in the node."""
        # Filter based on what is set / active?
        return dict([(v.name, v) for v in [self.cost]])

    def transform(self):
        """writeme"""
        ERROR_FMT = "Port error: '%s' not set."
        assert self.input.variable, ERROR_FMT % self.input
        assert self.weight.variable, ERROR_FMT % self.weight
        assert self.threshold.variable, ERROR_FMT % self.threshold
        var_max = T.max(self.input.variable - self.threshold.variable)
        self.cost.variable = FX.relu(var_max) * self.weight.variable


class MeanSquaredError(Loss):
    """writeme"""
    def __init__(self, name, weighted=False):
        # Input Validation
        Loss.__init__(self, name=name, weighted=weighted)
        self.prediction = core.Port(
            name=self.__own__("prediction"))
        self.target = core.Port(
            name=self.__own__("target"))
        self.weights = core.Port(name=self.__own__("weights"),
                                 shape=[]) if weighted else False

    @property
    def inputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        ports = [self.prediction, self.target]
        if self.weights:
            ports.append(self.weights)
        return dict([(v.name, v) for v in ports])

    def transform(self):
        """writeme"""
        assert self.prediction.variable, "Port error: 'prediction' not set."
        prediction = self.prediction.variable
        assert self.target.variable, "Port error: 'target' not set."
        target = self.target.variable

        squared_error = T.pow(prediction - target, 2.0).flatten(2)
        # self.loss.variable = T.mean(squared_error, axis=-1)
        self.loss.variable = T.sum(squared_error, axis=-1)
        if self.weights:
            self.loss.variable *= self.weights.variable
        self.cost.variable = T.mean(self.loss.variable)


class SparseMeanSquaredError(Loss):
    """writeme"""
    def __init__(self, name, weighted=False):
        # Input Validation
        Loss.__init__(self, name=name, weighted=weighted)
        self.prediction = core.Port(
            name=self.__own__("prediction"))
        self.target = core.Port(
            name=self.__own__("target"))
        self.index = core.Port(
            name=self.__own__("index"))
        self.weights = core.Port(name=self.__own__("weights"),
                                 shape=[]) if weighted else False

    @property
    def inputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        ports = [self.prediction, self.target, self.index]
        if self.weights:
            ports.append(self.weights)
        return dict([(v.name, v) for v in ports])

    def transform(self):
        """writeme"""
        assert self.prediction.variable, "Port error: 'prediction' not set."
        prediction = self.prediction.variable
        assert self.target.variable, "Port error: 'target' not set."
        target = self.target.variable
        assert self.index.variable, "Port error: 'index' not set."
        index = self.index.variable

        # Create the local givens.
        batch_idx = T.arange(index.shape[0], dtype='int32')
        sparse_prediction = prediction[batch_idx, index]
        self.loss.variable = T.pow(sparse_prediction - target, 2.0)
        if self.weights:
            self.loss.variable *= self.weights.variable
        self.cost.variable = T.mean(self.loss.variable)


class CrossEntropy(Loss):
    """writeme"""
    def __init__(self, name, weighted=False):
        # Input Validation
        Loss.__init__(self, name=name, weighted=weighted)
        self.prediction = core.Port(
            name=self.__own__("prediction"))
        self.target = core.Port(
            name=self.__own__("target"))
        self.weights = core.Port(name=self.__own__("weights"),
                                 shape=[]) if weighted else None

    @property
    def inputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        ports = [self.prediction, self.target]
        if self.weights:
            ports.append(self.weights)
        return dict([(v.name, v) for v in ports])

    def transform(self):
        """writeme"""
        assert self.prediction.variable, "Port error: 'prediction' not set."
        prediction = self.prediction.variable
        assert self.target.variable, "Port error: 'target' not set."
        target = self.target.variable

        loss = -T.mean(target * T.log(prediction)
                       + (1.0 - target) * T.log(1.0 - prediction), axis=1)
        self.loss.variable = loss
        if self.weights:
            loss *= self.weights.variable
        self.cost.variable = T.mean(loss)


class SparseCrossEntropy(Loss):
    """writeme"""
    def __init__(self, name, weighted=False):
        # Input Validation
        Loss.__init__(self, name=name, weighted=weighted)
        self.prediction = core.Port(
            name=self.__own__("prediction"))
        self.target = core.Port(
            name=self.__own__("target"))
        self.index = core.Port(
            name=self.__own__("index"))
        self.weights = core.Port(name=self.__own__("weights"),
                                 shape=[]) if weighted else None

    @property
    def inputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        ports = [self.prediction, self.target, self.index]
        if self.weights:
            ports.append(self.weights)
        return dict([(v.name, v) for v in ports])

    def transform(self):
        """writeme"""
        assert self.prediction.variable, "Port error: 'prediction' not set."
        prediction = self.prediction.variable
        assert self.target.variable, "Port error: 'target' not set."
        target = self.target.variable
        assert self.index.variable, "Port error: 'index' not set."
        index = self.index.variable

        batch_idx = T.arange(index.shape[0], dtype='int32')
        sparse_prediction = prediction[batch_idx, index]
        loss = -(target * T.log(sparse_prediction)
                     + (1.0 - target) * T.log(1.0 - sparse_prediction))
        self.loss.variable = loss
        if self.weights:
            loss *= self.weights.variable
        self.cost.variable = T.mean(loss)
