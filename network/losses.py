"""writeme."""

import theano.tensor as T
from . import core
from . import FLOATX
from . import functions


class Accumulator(list):

    def __init__(self, losses, variables):
        list.__init__(self, losses)
        self._total(variables)

    def _total(self, variables):
        self._inputs = dict()
        self._total_loss = 0.0
        for l in self:
            scalar_loss, inputs = l.loss(variables)
            self._inputs.update(inputs)
            self._total_loss += scalar_loss

    @property
    def total(self):
        return self._total_loss

    @property
    def inputs(self):
        return self._inputs


class Loss(core.JObject):

    def __init__(self, name, **kwargs):
        """writeme."""
        self.name = name
        self.__args__ = dict(**kwargs)

    def is_ready(self):
        """Return true when all input ports are loaded."""
        return all([p.variable for p in self.inputs.values()])

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

    def __own__(self, name):
        return "%s.%s" % (self.name, name)

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
        raise NotImplementedError("Subclass me")


class NegativeLogLikelihood(Loss):
    """writeme"""
    def __init__(self, name):
        # Input Validation
        Loss.__init__(self, name=name)
        self.likelihood = core.Port(
            name=self.__own__("likelihood"))
        self.target_idx = core.Port(
            name=self.__own__("target_idx"), shape=[])
        self.loss = core.Port(
            shape=[], name=self.__own__("loss"))
        self.cost = core.Port(
            shape=None, name=self.__own__("cost"))

    @property
    def inputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        return dict([(v.name, v) for v in [self.likelihood, self.target_idx]])

    @property
    def outputs(self):
        """Return a list of all active Outputs in the node."""
        # Filter based on what is set / active?
        return dict([(v.name, v) for v in [self.loss, self.cost]])

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


class ContrastiveDivergence(Loss):
    _DISTANCE = 'distance'
    _SCORE = 'score'
    _MARGIN = 'margin'

    def __init__(self, distance, score, margin):
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


class LpNorm(Loss):
    _VARIABLE = 'variable'
    _WEIGHT = 'weight'
    _P = 'p'

    def __init__(self, variable, weight, p):
        # Input Validation
        self.update(variable=variable, p=p, weight=weight)
        Loss.__init__(self)

    @property
    def _p(self):
        return self.get(self._P)

    def loss(self, variables):
        """
        variables : dict
            Set of URL-keyed variables from which to select.
        """
        variable = variables[self[self._VARIABLE]]
        scalar_loss = T.pow(T.sum(T.pow(T.abs_(variable), self._p)),
                            1.0 / self._p)
        weight = T.scalar(name=self[self._WEIGHT])
        return scalar_loss*weight, {weight.name: weight}


class L1Norm(Loss):
    _VARIABLE = 'variable'
    _WEIGHT = 'weight'

    def __init__(self, variable, weight):
        # Input Validation
        self.update(variable=variable, weight=weight)
        Loss.__init__(self)

    def loss(self, variables):
        """
        variables : dict
            Set of URL-keyed variables from which to select.
        """
        variable = variables[self[self._VARIABLE]]
        scalar_loss = T.sum(T.abs_(variable))
        weight = T.scalar(name=self[self._WEIGHT])
        return scalar_loss*weight, {weight.name: weight}


class L2Norm(Loss):
    _VARIABLE = 'variable'
    _WEIGHT = 'weight'

    def __init__(self, variable, weight):
        # Input Validation
        self.update(variable=variable, weight=weight)
        Loss.__init__(self)

    def loss(self, variables):
        """
        variables : dict
            Set of URL-keyed variables from which to select.
        """
        variable = variables[self[self._VARIABLE]]
        scalar_loss = T.pow(T.sum(T.pow(variable, 2.0)), 0.5)
        weight = T.scalar(name=self[self._WEIGHT])
        return scalar_loss*weight, {weight.name: weight}

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
