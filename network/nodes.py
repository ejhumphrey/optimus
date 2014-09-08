"""TODO(ejhumphrey): write me."""

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample

from . import core
from . import FLOATX
from . import functions


def compile(inputs, outputs):
    return theano.function(inputs=[x.variable for x in inputs],
                           outputs=[z.variable for z in outputs],
                           allow_input_downcast=True)


# --- Node Implementations ---
class Node(core.JObject):
    """
    Nodes in the graph perform parameter management and micro-math operations.
    """
    def __init__(self, name, **kwargs):
        """writeme."""
        self.name = name
        self.__args__ = dict(**kwargs)
        self.act_type = 'linear'
        self._numpy_rng = np.random.RandomState()
        self._theano_rng = RandomStreams(self._numpy_rng.randint(2 ** 30))
        self._inputs = []
        self._params = []
        self._outputs = []

    # --- Public Properties ---
    @property
    def activation(self):
        """TODO(ejhumphrey): write me."""
        return functions.Activations.get(self.act_type)

    def is_ready(self):
        """Return true when all input ports are loaded."""
        set_inputs = all([p.variable for p in self.inputs.values()])
        set_outputs = all([p.variable for p in self.outputs.values()])
        return set_inputs and not set_outputs

    def reset(self):
        """TODO(ejhumphrey): write me."""
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
        """TODO(ejhumphrey): write me."""
        return "%s.%s" % (self.name, name)

    # --- Subclassed methods ---
    def transform(self):
        """TODO(ejhumphrey): write me."""
        raise NotImplementedError("Subclass me!")

    @property
    def inputs(self):
        """Return a dict of all active Outputs in the node."""
        return dict([(v.name, v) for v in self._inputs])

    @property
    def params(self):
        """Return a dict of all Parameters in the node."""
        return dict([(v.name, v) for v in self._params])

    @property
    def outputs(self):
        """Return a dict of all active Outputs in the node."""
        return dict([(v.name, v) for v in self._outputs])


class Accumulate(Node):
    """Summation node."""
    def __init__(self, name):
        # Input Validation
        Node.__init__(self, name=name)
        self.input_list = core.PortList(name=self.__own__("input_list"))
        self._inputs.append(self.input_list)
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)

    def transform(self):
        """writeme"""
        assert self.is_ready(), "Not all ports are set."
        self.output.variable = sum(self.input_list.variable)


class Concatenate(Node):
    """

    """
    def __init__(self, name, axis=-1):
        Node.__init__(self, name=name, axis=axis)
        self.input_list = core.PortList(name=self.__own__("input_list"))
        self._inputs.append(self.input_list)
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)
        self.axis = axis

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        self.output.variable = T.concatenate(self.input_list.variable,
                                             axis=self.axis)


class Stack(Node):
    """Form a rank+1 tensor of a set of inputs; optionally reorder the axes."""
    def __init__(self, name, axes=None):
        Node.__init__(self, name=name, axes=axes)
        self.input_list = core.PortList(name=self.__own__("input_list"))
        self._inputs.append(self.input_list)
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)
        self.axes = axes

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        output = T.stack(*self.input_list.variable)
        if self.axes:
            output = T.transpose(output, axes=self.axes)
        self.output.variable = output


class Unary(Node):
    """Single input / output nodes."""
    def __init__(self, name, **kwargs):
        Node.__init__(self, name=name, **kwargs)
        self.input = core.Port(name=self.__own__('input'))
        self._inputs.append(self.input)
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)


class Dimshuffle(Unary):
    def __init__(self, name, axes):
        Unary.__init__(self, name=name, axes=axes)
        self.axes = axes

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        self.output.variable = self.input.variable.dimshuffle(*self.axes)


class Log(Unary):
    def __init__(self, name):
        Unary.__init__(self, name=name)

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        self.output.variable = T.log(self.input.variable)


class Sigmoid(Unary):
    def __init__(self, name):
        Unary.__init__(self, name=name)

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        self.output.variable = functions.sigmoid(self.input.variable)


class Softmax(Unary):
    """Apply the softmax to an input."""
    def __init__(self, name):
        Unary.__init__(self, name=name)

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        self.output.variable = T.nnet.softmax(self.input.variable)


class RectifiedLinear(Unary):
    """Apply the (hard) rectified linear function to an input."""
    def __init__(self, name):
        Unary.__init__(self, name=name)

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        self.output.variable = functions.relu(self.input.variable)


class Tanh(Unary):
    """Apply the hyperbolic tangent to an input."""
    def __init__(self, name):
        Unary.__init__(self, name=name)

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        self.output.variable = T.tanh(self.input.variable)


class Sum(Unary):
    """Returns the sum of an input, or over a given axis."""
    def __init__(self, name, axis=None):
        Unary.__init__(self, name=name, axis=axis)
        self.axis = axis

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        if self.axis is None:
            self.output.variable = T.sum(self.input.variable)
        else:
            self.output.variable = T.sum(self.input.variable, axis=self.axis)


class Mean(Unary):
    """Returns the mean of an input, or over a given axis."""
    def __init__(self, name, axis=None):
        Unary.__init__(self, name=name, axis=axis)
        self.axis = axis

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        if self.axis is None:
            self.output.variable = T.mean(self.input.variable)
        else:
            self.output.variable = T.mean(self.input.variable, axis=self.axis)


class Max(Unary):
    """Returns the max of an input, or over a given axis."""
    def __init__(self, name, axis=None):
        Unary.__init__(self, name=name, axis=axis)
        self.axis = axis

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        if self.axis is None:
            self.output.variable = T.max(self.input.variable)
        else:
            self.output.variable = T.max(self.input.variable, axis=self.axis)


class Min(Unary):
    """Returns the min of an input, or over a given axis."""
    def __init__(self, name, axis=None):
        Unary.__init__(self, name=name, axis=axis)
        self.axis = axis

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        if self.axis is None:
            self.output.variable = T.min(self.input.variable)
        else:
            self.output.variable = T.min(self.input.variable, axis=self.axis)


class Gain(Unary):
    """Apply a scalar gain to an input."""
    def __init__(self, name):
        Unary.__init__(self, name=name)
        self.weight = core.Parameter(shape=None, name=self.__own__('weight'))
        self._params.append(self.weight)

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        self.output.variable = self.weight.variable * self.input.variable


class Affine(Unary):
    """
    Affine Transform Layer
      (i.e., a fully-connected non-linear projection)

    """
    def __init__(self, name, input_shape, output_shape, act_type):
        Unary.__init__(
            self,
            name=name,
            input_shape=input_shape,
            output_shape=output_shape,
            act_type=act_type)
        self.act_type = act_type

        # TODO(ejhumphrey): This is super important but kind of a hack. Think
        #   on this and come up with something better.
        self.input.shape = input_shape
        self.output.shape = output_shape

        n_in = int(np.prod(input_shape[1:]))
        n_out = int(np.prod(output_shape[1:]))
        weight_shape = [n_in, n_out]

        self.weights = core.Parameter(
            shape=weight_shape, name=self.__own__('weights'))
        self.bias = core.Parameter(
            shape=[n_out], name=self.__own__('bias'))
        self._params.extend([self.weights, self.bias])
        self.dropout = None

    def enable_dropout(self):
        self.dropout = core.Port(shape=None, name=self.__own__('dropout'))
        self._inputs.append(self.dropout)

    def disable_dropout(self):
        if self.dropout:
            self._inputs.remove(self.dropout)
        self.dropout = None

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        weights = self.weights.variable
        bias = self.bias.variable.dimshuffle('x', 0)

        x_in = T.flatten(self.input.variable, outdim=2)
        z_out = self.activation(T.dot(x_in, weights) + bias)
        if self.dropout:
            print "Performing dropout in %s" % self.name
            dropout = self.dropout.variable
            selector = self._theano_rng.binomial(
                size=self.bias.shape, p=1.0 - dropout)
            # Scale up by the ratio of the number of units that are 'off'.
            z_out *= selector.dimshuffle('x', 0) / (1.0 - dropout)

        output_shape = list(self.output.shape)[1:]
        self.output.variable = T.reshape(
            z_out, [z_out.shape[0]] + output_shape)


class Conv3D(Unary):
    """TODO(ejhumphrey): write me."""

    def __init__(self, name, input_shape, weight_shape,
                 pool_shape=(1, 1),
                 downsample_shape=(1, 1),
                 act_type='relu',
                 border_mode='valid'):
        """

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data, as (in_maps, in_dim0, in_dim1).
        weight_shape : tuple
            Shape for all kernels, as (num_kernels, w_dim0, w_dim1).
        pool_shape : tuple, default=(1,1)
            2D tuple to pool over each feature map, as (p_dim0, p_dim1).
        downsample_shape : tuple, default=(1,1)
            2D tuple for downsampling each feature map, as (p_dim0, p_dim1).
        act_type : str, default='relu'
            Name of the activation function to use.
        border_mode : str, default='valid'
            Convolution method for dealing with the edge of a feature map.

        """
        Unary.__init__(
            self,
            name=name,
            input_shape=input_shape,
            weight_shape=weight_shape,
            pool_shape=pool_shape,
            downsample_shape=downsample_shape,
            border_mode=border_mode,
            act_type=act_type)

        # Make sure the weight_shape argument is formatted properly.
        w_shp = list(weight_shape)
        if len(w_shp) == 3:
            w_shp.insert(1, input_shape[1])
        elif len(w_shp) == 4 and w_shp[1] is None:
            w_shp[1] = input_shape[1]
        elif len(w_shp) == 4:
            assert w_shp[1] == input_shape[1], \
                "weight_shape[1] must align with input_shape[1]: " \
                "%d!=%d." % (w_shp[1], input_shape[1])
        else:
            raise ValueError("'weight_shape' must be length 3 or 4.")
        weight_shape = tuple(w_shp)

        d0_in, d1_in = input_shape[-2:]
        if border_mode == 'valid':
            d0_out = int(d0_in - weight_shape[-2] + 1)
            d0_out /= pool_shape[0]
            d1_out = int(d1_in - weight_shape[-1] + 1)
            d1_out /= pool_shape[1]
        elif border_mode == 'same':
            d0_out, d1_out = d0_in, d1_in
        elif border_mode == 'full':
            """TODO(ejhumphrey): Implement full-convolution math."""
            raise NotImplementedError("Haven't implemented 'full' shape yet.")

        output_shape = (input_shape[0], weight_shape[0], d0_out, d1_out)

        # TODO(ejhumphrey): This is super important but kind of a hack. Think
        #   on this and come up with something better.
        self.input.shape = input_shape
        self.output.shape = output_shape

        self.dropout = None
        self.weights = core.Parameter(
            shape=weight_shape,
            name=self.__own__('weights'))
        self.bias = core.Parameter(
            shape=weight_shape[:1],
            name=self.__own__('bias'))
        self._params.extend([self.weights, self.bias])

        self.pool_shape = pool_shape
        self.downsample_shape = downsample_shape
        self.border_mode = border_mode

        # Param init
        fan_in = np.prod(self.weights.shape[1:])
        weight_values = self._numpy_rng.normal(
            loc=0.0, scale=np.sqrt(3. / fan_in),
            size=self.weights.shape)

        if act_type == 'sigmoid':
            weight_values *= 4

        self.weights.value = weight_values.astype(FLOATX)

    def transform(self):
        """writeme."""
        assert self.is_ready(), "Input ports not set."
        weights = self.weights.variable
        bias = self.bias.variable.dimshuffle('x', 0, 'x', 'x')
        output = T.nnet.conv.conv2d(
            input=self.input.variable,
            filters=weights,
            filter_shape=self.weights.shape,
            border_mode=self.border_mode)

        output = self.activation(output + bias)

        if self.dropout:
            output_shape = list(self.output.shape)
            dropout = self.dropout.variable
            selector = self._theano_rng.binomial(
                size=output_shape,
                p=1.0 - dropout)

            output *= selector.dimshuffle('x', 0, 'x', 'x') / (1.0 - dropout)

        output = downsample.max_pool_2d(
            output, self.pool_shape, ignore_border=False)
        self.output.variable = output


class RadialBasis(Unary):
    """Radial Basis Layer, i.e. L2-Distance, with internal weights."""
    def __init__(self, name, input_shape, output_shape):
        Unary.__init__(
            self,
            name=name,
            input_shape=input_shape,
            output_shape=output_shape)

        # TODO(ejhumphrey): This is super important but kind of a hack. Think
        #   on this and come up with something better.
        self.input.shape = input_shape
        self.output.shape = output_shape

        n_in = int(np.prod(input_shape[1:]))
        n_out = int(np.prod(output_shape[1:]))
        weight_shape = [n_in, n_out]
        self.weights = core.Parameter(
            shape=weight_shape, name=self.__own__('weights'))
        self._params.append(self.weights)

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        weights = self.weights.variable

        x_in = T.flatten(self.input.variable, outdim=2)
        z_out = T.pow(T.abs_(x_in - weights), 2.0).sum(axis=1)

        output_shape = list(self.output.shape)[1:]
        self.output.variable = T.reshape(
            z_out, [z_out.shape[0]] + output_shape)


class Conv2D(Node):
    """TODO(ejhumphrey): Implement me."""

    def __init__(self, layer_args):
        """
        layer_args : ConvArgs

        """
        raise NotImplementedError("come back to this")
        Node.__init__(self, layer_args)

        # Create all the weight values at once
        weight_shape = self.param_shapes.get("weights")
        fan_in = np.prod(weight_shape[1:])
        weights = self.numpy_rng.normal(loc=0.0,
                                        scale=np.sqrt(3. / fan_in),
                                        size=weight_shape)

        if self.get("activation") == 'sigmoid':
            weights *= 4

        bias = np.zeros(weight_shape[0])
        self.param_values = {self.__own__('weights'): weights,
                             self.__own__('bias'): bias, }

    def transform(self, x_in):
        """writeme"""
        raise NotImplementedError("come back to this")
        W = self._theta['weights']
        b = self._theta['bias']
        weight_shape = self.param_shapes.get("weights")
        z_out = T.nnet.conv.conv2d(input=x_in,
                                   filters=W,
                                   filter_shape=weight_shape,
                                   border_mode=self.get("border_mode"))

        selector = self.theano_rng.binomial(size=self.output_shape[:1],
                                            p=1.0 - self.dropout,
                                            dtype=FLOATX)

        z_out = self.activation(z_out + b.dimshuffle('x', 0, 'x', 'x'))
        z_out *= selector.dimshuffle('x', 0, 'x', 'x') * (self.dropout + 0.5)
        return downsample.max_pool_2d(
            z_out, self.get("pool_shape"), ignore_border=False)


class CrossProduct(Node):
    """
    Affine Transform Layer
      (i.e., a fully-connected non-linear projection)

    """
    def __init__(self, name):
        Node.__init__(self, name=name)
        self.input_a = core.Port(name=self.__own__('input_a'))
        self.input_b = core.Port(name=self.__own__('input_b'))
        self.output = core.Port(name=self.__own__('output'))

    @property
    def inputs(self):
        """Return a dict of all active Inputs in the node."""
        # TODO(ejhumphrey@nyu.edu): Filter based on what is set / active?
        # i.e. dropout yes/no?
        ports = [self.input_a, self.input_b]
        return dict([(v.name, v) for v in ports])

    @property
    def params(self):
        """Return a dict of all Parameters in the node."""
        # Filter based on what is set / active?
        return {}

    @property
    def outputs(self):
        """Return a dict of all active Outputs in the node."""
        # Filter based on what is set / active?
        return {self.output.name: self.output}

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."

        in_a = self.input_a.variable.dimshuffle(0, 1, 'x')
        in_b = self.input_b.variable.dimshuffle(0, 'x', 1)

        self.output.variable = (in_a * in_b).flatten(2)


class Normalize(Unary):
    """

    """
    def __init__(self, name, mode='l2', scale_factor=1.0):
        Unary.__init__(self, name=name, mode=mode)
        self.input = core.Port(name=self.__own__('input'))
        self.output = core.Port(name=self.__own__('output'))
        self.mode = mode
        self.scale_factor = scale_factor

    def transform(self):
        """In-place transformation"""
        assert self.is_ready(), "Not all ports are set."
        input_var = self.input.variable.flatten(2)

        if self.mode == 'l1':
            scalar = T.sum(T.abs_(input_var), axis=1)
        elif self.mode == 'l2':
            scalar = T.sqrt(T.sum(T.abs_(input_var)**2.0, axis=1))

        scalar += 1.0 * T.eq(scalar, 0)
        new_shape = [0] + ['x']*(self.input.variable.ndim - 1)
        scalar = scalar.dimshuffle(*new_shape)
        self.output.variable = self.scale_factor * self.input.variable / scalar


class SelectIndex(Node):
    """writeme"""
    def __init__(self, name):
        # Input Validation
        Node.__init__(self, name=name)
        self.input = core.Port(name=self.__own__("input"))
        self.index = core.Port(name=self.__own__("index"), shape=[])
        self._inputs.extend([self.input, self.index])
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)

    def transform(self):
        """writeme"""
        assert self.is_ready()
        assert self.input.variable.ndim == 2
        col_index = self.index.variable
        row_index = T.arange(col_index.shape[0], dtype='int32')
        self.output.variable = self.input.variable[row_index, col_index]


class SquaredEuclidean(Node):
    """Squared Euclidean Node

    Computes: z_n = \sum_i(xA_n[i] - xB_n[i])^2
    """
    def __init__(self, name):
        """

        """
        Node.__init__(self, name=name)
        self.input_a = core.Port(name=self.__own__("input_a"))
        self.input_b = core.Port(name=self.__own__("input_b"))
        self._inputs.extend([self.input_a, self.input_b])
        self.output = core.Port(name=self.__own__('output'))
        self._outputs.append(self.output)

    def transform(self):
        """Transform inputs to outputs."""
        assert self.is_ready()
        if self.input_a.variable.ndim >= 2:
            xA = T.flatten(self.input_a.variable, outdim=2)
            xB = T.flatten(self.input_b.variable, outdim=2)
            axis = 1
        else:
            xA = self.input_a.variable
            xB = self.input_b.variable
            axis = None
        self.output.variable = T.pow(xA - xB, 2.0).sum(axis=axis)


class L1Magnitude(Unary):
    def __init__(self, name, axis=None):
        Unary.__init__(self, name=name, axis=None)
        self.axis = axis

    def transform(self):
        """writeme"""
        assert self.is_ready()
        self.output.variable = T.sum(T.abs_(self.input.variable),
                                     axis=self.axis)


class L2Magnitude(Unary):
    def __init__(self, name, axis=None):
        Unary.__init__(self, name=name, axis=None)
        self.axis = axis

    def transform(self):
        """writeme"""
        assert self.is_ready()
        self.output.variable = T.sqrt(T.sum(T.pow(self.input.variable, 2.0),
                                            axis=self.axis))
