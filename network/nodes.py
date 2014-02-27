"""writeme
"""

import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample

from . import core
from . import FLOATX
from . import functions


def NodeFactory(args):
    """writeme."""
    assert "cls" in args, "Must contain a key for 'cls'"
    return dict(**args)
    # return eval(args.pop('cls'))(**args)


# --- Node Implementations ------
class Node(core.Struct):
    """
    Nodes in the graph perform parameter management and micro-math operations.
    """
    def __init__(self, name):
        """writeme."""
        self.name = name
        self.act_type = 'linear'
        self.inputs = core.Struct()
        self.outputs = core.Struct()
        self.params = core.Struct()
        self.scalars = core.Struct()

        self._numpy_rng = np.random.RandomState()
        self._theano_rng = RandomStreams(self._numpy_rng.randint(2 ** 30))

    # --- Public Properties ---
    @property
    def type(self):
        """writeme"""
        return self.__class__.__name__

    @property
    def activation(self):
        """writeme"""
        return functions.Activations.get(self.act_type)

    def transform(self):
        """writeme"""
        raise NotImplementedError("Subclass me!")


class Affine(Node):
    """
    Affine Transform Layer
      (i.e., a fully-connected non-linear projection)

    """
    def __init__(self, name, inputs, outputs, params, scalars, act_type):
        Node.__init__(self, name)
        self.act_type = act_type
        self.inputs.x_in = core.Port(**inputs['x_in'])
        self.outputs.z_out = core.Port(**outputs['z_out'])
        self.params.weights = core.Parameter(**params['weights'])
        self.params.bias = core.Parameter(**params['bias'])
        if 'dropout' in scalars:
            self.scalars.dropout = core.Scalar()

    def transform(self, x_in):
        """
        will fix input tensors to be matrices as the following:
        (N x d0 x d1 x ... dn) -> (N x prod(d_(0:n)))

        Parameters
        ----------
        inputs: dict
            Must contain all known data inputs to this node, keyed by full
            URLs. Will fail loudly otherwise.

        Returns
        -------
        outputs: dict
            Will contain all outputs generated by this node, keyed by full
            name. Note that the symbolic outputs will take this full name
            internal to each object.
        """
        weights = self.params.weights.variable
        bias = self.params.bias.variable.dimshuffle('x', 0)

        x_in = T.flatten(x_in, outdim=2)
        z_out = self.activation(T.dot(x_in, weights) + bias)
        output_shape = list(self.outputs.z_out.shape)
        z_out = T.reshape(z_out, [z_out.shape[0]] + output_shape)
        if 'dropout' in self.scalars.keys():
            dropout = self.scalars.dropout.variable
            selector = self._theano_rng.binomial(
                size=output_shape, p=1.0 - dropout)
            z_out *= selector.dimshuffle('x', 0) * (dropout + 0.5)

        return z_out

    @classmethod
    def simple(cls, name, input_shape, output_shape, act_type,
               enable_dropout=False):
        """writeme"""
        n_in = int(np.prod(input_shape))
        n_out = int(np.prod(output_shape))

        inputs = dict(x_in=core.Port(input_shape))
        outputs = dict(z_out=core.Port(output_shape))
        params = dict(weights=core.Parameter([n_in, n_out]),
                      bias=core.Parameter([n_out, ]))
        scalars = dict()
        if enable_dropout:
            scalars.update(dropout=core.Scalar())

        return cls(name=name, inputs=inputs, outputs=outputs, params=params,
                   scalars=scalars, act_type=act_type)


class Conv3D(Node):
    """ (>^.^<) """

    def __init__(self, name, inputs, outputs, params, scalars, act_type,
                 pool_shape, downsample_shape, border_mode):
        Node.__init__(self, name)
        self.act_type = act_type
        self.inputs.x_in = core.Port(**inputs['x_in'])
        self.outputs.z_out = core.Port(**outputs['z_out'])
        self.params.weights = core.Parameter(**params['weights'])
        self.params.bias = core.Parameter(**params['bias'])
        if 'dropout' in scalars:
            self.scalars.dropout = core.Scalar()

        self.pool_shape = pool_shape
        self.downsample_shape = downsample_shape
        self.border_mode = border_mode

        fan_in = np.prod(self.params.weights.shape[1:])
        weight_values = self._numpy_rng.normal(
            loc=0.0, scale=np.sqrt(3. / fan_in),
            size=self.params.weights.shape)

        if act_type == 'sigmoid':
            weight_values *= 4

        self.params.weights.value = weight_values

    @classmethod
    def simple(cls, name, input_shape, weight_shape,
               pool_shape=(1, 1),
               downsample_shape=(1, 1),
               act_type='relu',
               border_mode='valid',
               enable_dropout=False):
        """Convenience Constructor

        Parameters
        ----------
        name: str
            Name for this node.
        input_shape : tuple
            Shape of the input data, as (in_maps, in_dim0, in_dim1).
        weight_shape : tuple
            Shape for all kernels, as (num_kernels, w_dim0, w_dim1).
        pool_shape : tuple
            2D tuple to pool over each feature map, as (p_dim0, p_dim1).
        downsample_shape : tuple
            2D tuple for downsampling each feature map, as (p_dim0, p_dim1).
        act_type : str
            Name of the activation function to use.
        border_mode : str
            Convolution method for dealing with the edge of a feature map.

        """
        # Make sure the weight_shape argument is formatted properly.
        w = list(weight_shape)
        if len(w) == 3:
            w.insert(1, input_shape[0])
        elif len(w) == 4:
            assert w[1] == input_shape[0], \
                "weight_shape[1] must align with input_shape[0]: " \
                "%d!=%d." % (w[1], input_shape[0])
        else:
            raise ValueError("'weight_shape' must be length 3 or 4.")
        weight_shape = tuple(w)

        d0_in, d1_in = input_shape[-2:]
        if border_mode == 'valid':
            d0_out = int(d0_in - weight_shape[-2] + 1)
            d0_out /= pool_shape[0]
            d1_out = int(d1_in - weight_shape[-1] + 1)
            d1_out /= pool_shape[1]
        elif border_mode == 'same':
            d0_out, d1_out = d0_in, d1_in
        elif border_mode == 'full':
            raise NotImplementedError("Haven't implemented 'full' shape yet.")

        output_shape = (weight_shape[0], d0_out, d1_out)
        scalars = dict()
        if enable_dropout:
            scalars.update(dropout=core.Scalar())

        return cls(name=name,
                   inputs=dict(x_in=core.Port(input_shape)),
                   outputs=dict(z_out=core.Port(output_shape)),
                   params=dict(weights=core.Parameter(weight_shape),
                               bias=core.Parameter(weight_shape[:1])),
                   scalars=scalars,
                   act_type=act_type,
                   downsample_shape=downsample_shape,
                   pool_shape=pool_shape,
                   border_mode=border_mode)

    def transform(self, x_in):
        """writeme."""
        weights = self.params.weights.variable
        bias = self.params.bias.variable.dimshuffle('x', 0, 'x', 'x')
        z_out = T.nnet.conv.conv2d(
            input=x_in,
            filters=weights,
            filter_shape=self.params.weights.shape,
            border_mode=self.border_mode)

        z_out = self.activation(z_out + bias)

        if 'dropout' in self.scalars.keys():
            output_shape = list(self.outputs.z_out.shape)
            dropout = self.scalars.dropout.variable
            selector = self._theano_rng.binomial(
                size=output_shape,
                p=1.0 - dropout)

            z_out *= selector.dimshuffle('x', 0, 'x', 'x') * (dropout + 0.5)

        z_out = downsample.max_pool_2d(
            z_out, self.pool_shape, ignore_border=False)
        return z_out


class Softmax(Affine):
    """writeme. """

    @classmethod
    def simple(cls, name, input_shape, n_out, act_type):
        """
        """
        n_in = int(np.prod(input_shape))
        inputs = dict(x_in=core.Port(input_shape))
        outputs = dict(z_out=core.Port([n_out]))
        params = dict(weights=core.Parameter([n_in, n_out]),
                      bias=core.Parameter([n_out, ]))
        return cls(name=name, inputs=inputs, outputs=outputs, params=params,
                   scalars=dict(), act_type=act_type)

    def transform(self, x_in):
        """
        will fix input tensors to be matrices as the following:
        (N x d0 x d1 x ... dn) -> (N x prod(d_(0:n)))
        """
        weights = self.params.weights.variable
        bias = self.params.bias.variable.dimshuffle('x', 0)
        x_in = T.flatten(x_in, outdim=2)
        z_out = T.nnet.softmax(self.activation(T.dot(x_in, weights) + bias))
        return z_out


class Conv2D(Node):
    """ . """

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
        self.param_values = {self.own('weights'): weights,
                             self.own('bias'): bias, }

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


class LpDistance(Node):
    """
    Distance Node

    """
    _INPUT_A = 'input-A'
    _INPUT_B = 'input-B'
    _OUTPUT = 'output'
    _P = "p"

    def __init__(self, name, p=2.0):
        """

        """
        raise NotImplementedError("come back to this")
        self.update(dict(p=p))
        Node.__init__(self, name, act_type='linear')
        self._scalars.clear()

    # --- Over-ridden private properties ---
    @property
    def _input_shapes(self):
        """
        Returns
        -------
        shapes : dict
        """
        return {self._INPUT_A: (), self._INPUT_B: ()}

    @property
    def _output_shapes(self):
        """
        Returns
        -------
        shapes : dict
        """
        return {self._OUTPUT: ()}

    @property
    def _param_shapes(self):
        return {}

    def transform(self, inputs):
        """
        Parameters
        ----------
        inputs: dict
            Must contain all known data inputs to this node, keyed by full
            URLs. Will fail loudly otherwise.

        Returns
        -------
        outputs: dict
            Will contain all outputs generated by this node, keyed by full
            name. Note that the symbolic outputs will take this full name
            internal to each object.
        """
        assert self.validate_inputs(inputs)

        xA = T.flatten(inputs.get(self._own(self._INPUT_A)), outdim=2)
        xB = T.flatten(inputs.get(self._own(self._INPUT_B)), outdim=2)
        p = self.get(self._P)
        z_out = T.pow(T.pow(T.abs_(xA - xB), p).sum(axis=1), 1.0 / p)
        z_out.name = self.outputs[0]
        return {z_out.name: z_out}


class LpPenalty(Node):
    pass


