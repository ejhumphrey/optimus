"""Processing objects typically used as losses.

These classes are quite similar to Nodes (subclasses, in fact), with the
important distinction that each object has a singular scalar output.
"""

import theano.tensor as T

import optimus.core as core
from optimus.nodes import Node
import optimus.functions as functions


class NegativeLogLikelihood(Node):
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
        self.validate_ports()
        assert self.likelihoods.variable.ndim == 2

        col_index = self.index.variable
        row_index = T.arange(col_index.shape[0], dtype='int32')
        self.output.variable = -T.mean(T.log(
            self.likelihoods.variable[row_index, col_index]))


class CrossEntropy(Node):
    """Pointwise cross-entropy between a `prediction` and `target`.

    NOTE: Both inputs *must* be non-negative, and only `target` may contain
    zeros. Expect all hell to break loose if this is violated.
    """
    def __init__(self, name, epsilon=10.0**-6):
        # Input Validation
        Node.__init__(self, name=name, epsilon=epsilon)
        self.prediction = core.Port(name=self.__own__("prediction"))
        self.target = core.Port(name=self.__own__("target"))
        self._inputs.extend([self.prediction, self.target])
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)
        self.epsilon = epsilon

    def transform(self):
        """writeme"""
        self.validate_ports()

        prediction = self.prediction.variable
        target = self.target.variable
        eps_p1 = (1.0 + self.epsilon)
        output = target * T.log((prediction + self.epsilon) / eps_p1)
        output += (1.0 - target) * T.log(
            (1.0 - prediction + self.epsilon) / eps_p1)
        self.output.variable = -T.mean(output)


class MeanSquaredError(Node):
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
        self.validate_ports()
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
        self.validate_ports()
        x_in = self.input.variable.flatten(2)
        w_mag = T.sqrt(T.sum(T.pow(x_in, 2.0), axis=-1))
        self.output.variable = self.weight.variable * T.mean(w_mag)


class ContrastiveMargin(Node):
    """

    Inputs
    ------
    distance : vector
        Observed distance between datapoints.

    equivalence : vector
        Similarity scores, normalized to [0, 1], corresponding to least and
        most similar, respectively.

    sim_margin : scalar
        Margin between similar points within which no penalty is incurred.

    diff_margin : scalar
        Margin between dissimilar points within which no penalty is incurred.

    Outputs
    -------
    output : scalar
        Cost incurred given the input parameters.

    Equation
    --------
    Given D: distance, y: equivalence ...
        sim_cost = y*hwr(D - sim_margin)^2
        diff_cost = (1 - y) * hwr(diff_margin - D)^2
        total = ave(sim_cost + diff_cost)

    """
    def __init__(self, name):
        super(ContrastiveMargin, self).__init__(name=name)
        self.distance = core.Port(name=self.__own__('distance'))
        self.equivalence = core.Port(name=self.__own__('equivalence'))
        self.sim_margin = core.Port(name=self.__own__('sim_margin'))
        self.diff_margin = core.Port(name=self.__own__('diff_margin'))
        self._inputs.extend([self.distance, self.equivalence,
                             self.sim_margin, self.diff_margin])
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)

    def transform(self):
        """Transform inputs to outputs."""
        self.validate_ports()

        # TODO: Find a more reusable way of enforcing this behavior.
        if self.distance.variable.ndim != 1:
            raise ValueError("`distance` must be a vector.")

        if self.equivalence.variable.ndim != 1:
            raise ValueError("`equivalence` must be a vector.")

        dist = self.distance.variable
        equiv = self.equivalence.variable
        smarg = self.sim_margin.variable
        dmarg = self.diff_margin.variable

        sim_cost = T.pow(functions.relu(dist - smarg), 2.0)
        diff_cost = T.pow(functions.relu(dmarg - dist), 2.0)
        total_cost = equiv * sim_cost + (1 - equiv) * diff_cost
        self.output.variable = T.mean(total_cost)
