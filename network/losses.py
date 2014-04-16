"""writeme."""

import theano.tensor as T
from . import core
from . import FLOATX
from . import functions


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


class Accumulator(Loss):
    """writeme"""
    def __init__(self, name):
        # Input Validation
        Loss.__init__(self, name=name)
        self.input_list = core.PortList(
            name=self.__own__("input_list"))

    @property
    def inputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        return dict([(v.name, v) for v in [self.input_list]])

    def transform(self):
        """writeme"""
        assert self.input_list.variable, "Port error: 'input_list' not set."
        self.cost.variable = sum(self.input_list.variable)


class NegativeLogLikelihood(Loss):
    """writeme"""
    def __init__(self, name):
        # Input Validation
        Loss.__init__(self, name=name)
        self.likelihood = core.Port(
            name=self.__own__("likelihood"))
        self.target_idx = core.Port(
            name=self.__own__("target_idx"), shape=[])

    @property
    def inputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        return dict([(v.name, v) for v in [self.likelihood, self.target_idx]])

    # @property
    # def outputs(self):
    #     """Return a list of all active Outputs in the node."""
    #     # Filter based on what is set / active?
    #     return dict([(v.name, v) for v in [self.loss, self.cost]])

    def transform(self):
        """writeme"""
        assert self.likelihood.variable, "Port error: 'likelihood' not set."
        likelihood = self.likelihood.variable
        assert self.target_idx.variable, "Port error: 'target_idx' not set."
        target_idx = self.target_idx.variable
        # Create the local givens.
        batch_idx = T.arange(target_idx.shape[0], dtype='int32')
        self.loss.variable = -T.log(likelihood)[batch_idx, target_idx]
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
        return T.pow(soft_hinge(margin, x), 2.0)


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


# def mean_squared_error(name, inputs):
#     """
#     Returns
#     -------
#     scalar_loss : symbolic scalar
#         Cost of this penalty
#     inputs : dict
#         Dictionary of full param names and symbolic parameters.
#     """
#     INPUT_KEY = 'prediction'
#     assert INPUT_KEY in inputs, \
#         "Function expected a key named '%s' in 'inputs'." % INPUT_KEY
#     target = T.matrix(name=urls.append_param(name, 'target'))
#     raise NotImplementedError("Haven't finished this yet.")


# def l2_penalty(name, inputs):
#     """
#     Returns
#     -------
#     scalar_loss : symbolic scalar
#         Cost of this penalty
#     inputs : dict
#         Dictionary of full param names and symbolic parameters.
#     """
#     INPUT_KEY = 'input'
#     assert INPUT_KEY in inputs, \
#         "Function expected a key named '%s' in 'inputs'." % INPUT_KEY
#     hyperparam_name = urls.append_param(name, 'l2_penalty')
#     weight_decay = T.scalar(hyperparam_name, dtype=FLOATX)
#     scalar_loss = weight_decay * T.sum(T.pow(inputs[INPUT_KEY], 2.0))
#     return scalar_loss, {weight_decay.name: weight_decay}


# def l1_penalty(x_input):
#     """
#     Returns
#     -------
#     scalar_loss : symbolic scalar
#         Cost of this penalty
#     inputs : dict
#         Dictionary of full param names and symbolic parameters.
#     """
#     hyperparam_name = os.path.join(x_input.name, 'l1_penalty')
#     sparsity = T.scalar(hyperparam_name, dtype=FLOATX)
#     scalar_loss = sparsity * T.sum(T.abs_(x_input))
#     return scalar_loss, [sparsity]
