"""Processing objects typically used as losses.

These classes are quite similar to Nodes (subclasses, in fact), with the
important distinction that each object has a singular scalar output.
"""

import theano.tensor as T
from . import core
from .nodes import Node


class NegativeLogLikelihoodLoss(Node):
    """Indexed negative log-likelihood loss, i.e. for 1-of-k classifiers.

    In numpy syntax, computes the following:
        output = mean(-log(likelihoods[:, index]))

    The provided likelihoods must be 2D, and should satisfy (likelihoods > 0);
    life will be very miserable if the second condition is violated, but no
    checks are performed in-line.

    See also: optimus.nodes.SelectIndex
    """
    def __init__(self, name):
        # Input Validation
        Node.__init__(self, name=name)
        self.likelihoods = core.Port(name=self.__own__("likelihoods"))
        self.index = core.Port(name=self.__own__("index"), shape=[])
        self._inputs.extend([self.likelihoods, self.index])
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)

    def transform(self):
        """writeme"""
        assert self.is_ready()
        assert self.likelihoods.variable.ndim == 2

        col_index = self.index.variable
        row_index = T.arange(col_index.shape[0], dtype='int32')
        self.output.variable = -T.mean(T.log(
            self.likelihoods.variable[row_index, col_index]))


class CrossEntropyLoss(Node):
    """Pointwise cross-entropy between a `prediction` and `target`.

    NOTE: Both inputs *must* be non-negative, and only `target` may contain
    zeros. Expect all hell to break loose if this is violated.
    """
    def __init__(self, name):
        # Input Validation
        Node.__init__(self, name=name)
        self.prediction = core.Port(name=self.__own__("prediction"))
        self.target = core.Port(name=self.__own__("target"))
        self._inputs.extend([self.prediction, self.target])
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)

    def transform(self):
        """writeme"""
        assert self.is_ready()

        prediction = self.prediction.variable
        target = self.target.variable

        output = target * T.log(prediction)
        output += (1.0 - target) * T.log(1.0 - prediction)
        self.output.variable = -T.mean(output)


class MeanSquaredErrorLoss(Node):
    """Compute the mean squared error between a `prediction` and a `target`.

    See also: optimus.nodes.SquaredEuclidean
    """
    def __init__(self, name):
        # Input Validation
        Node.__init__(self, name=name)
        self.prediction = core.Port(name=self.__own__("prediction"))
        self.target = core.Port(name=self.__own__("target"))
        self._inputs.extend([self.prediction, self.target])
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)

    def transform(self):
        """writeme"""
        assert self.is_ready()
        if self.prediction.variable.ndim >= 2:
            xA = T.flatten(self.prediction.variable, outdim=2)
            xB = T.flatten(self.target.variable, outdim=2)
            axis = 1
        else:
            xA = self.prediction.variable
            xB = self.target.variable
            axis = None
        self.output.variable = T.mean(T.pow(xA - xB, 2.0).sum(axis=axis))


class WeightDecayPenalty(Node):
    """Compute the mean L2-magnitude of an `input`, scaled by `weight`.

    See also: optimus.nodes.L2Magnitude.
    """
    def __init__(self, name):
        Node.__init__(self, name=name)
        self.input = core.Port(name=self.__own__('input'))
        self.weight = core.Port(name=self.__own__('weight'))
        self._inputs.extend([self.input, self.weight])
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)

    def transform(self):
        assert self.is_ready(), "Not all ports are set."
        x_in = self.input.variable.flatten(2)
        w_mag = T.sqrt(T.sum(T.pow(x_in, 2.0), axis=-1))
        self.output.variable = self.weight.variable * T.mean(w_mag)
