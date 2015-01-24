"""Tests for Node objects."""

import unittest

import numpy as np
import optimus.nodes as nodes
import optimus.core as core


def __relu__(x):
    "Numpy Rectified Linear Unit."
    return 0.5*(np.abs(x) + x)


class NodeTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_Node(self):
        pass

    def test_Constant(self):
        n = nodes.Constant(name='test', shape=None)
        n.data.value = 1.0

        n.transform()
        fx = nodes.compile(inputs=[], outputs=[n.output])

        np.testing.assert_equal(fx()[0], 1.0)

    def test_Add(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        x2 = core.Input(name='x2', shape=(2, 2))

        n = nodes.Add(name='accumulate', num_inputs=2)
        n.input_0.connect(x1)

        with self.assertRaises(AssertionError):
            n.transform()

        n.input_1.connect(x2)
        self.assertIsNone(n.output.shape)
        n.transform()
        self.assertEqual(n.output.shape, (2, 2))

        fx = nodes.compile(inputs=[x1, x2],
                           outputs=[n.output])
        a = np.array([[3, -1], [3, 7]])
        b = np.array([[1, 2], [3, 4]])

        z = fx(a, b)[0]
        np.testing.assert_equal(z, np.array([[4, 1], [6, 11]]))

    @unittest.skip("Not fully implemented yet.")
    def test_Bincount(self):
        x1 = core.Input(name='x1', shape=(None,))

        n = nodes.Bincount(name='counter', max_int=3)
        n.input.connect(x1)
        n.transform()

        fx = nodes.compile(inputs=[x1], outputs=[n.counts])
        a = np.array([3, 0, 3, 1])

        np.testing.assert_equal(n.counts.value, np.array([0, 0, 0, 0]))
        np.testing.assert_equal(fx(a)[0], np.array([1, 1, 0, 2]))
        np.testing.assert_equal(fx(a)[0], np.array([2, 2, 0, 4]))

    def test_Concatenate(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        x2 = core.Input(name='x2', shape=(2, 2))
        a = np.array([[3, -1], [3, 7]])
        b = np.array([[1, 2], [3, 4]])

        for axis in range(2):
            n = nodes.Concatenate(name='concatenate', num_inputs=2, axis=axis)
            n.input_0.connect(x1)
            with self.assertRaises(AssertionError):
                n.transform()
            n.input_1.connect(x2)
            n.transform()

            fx = nodes.compile(inputs=[x1, x2],
                               outputs=[n.output])

            z = fx(a, b)[0]
            np.testing.assert_equal(z, np.concatenate([a, b], axis=axis))

    def test_Stack(self):
        x1 = core.Input(name='x1', shape=(2, 3))
        x2 = core.Input(name='x2', shape=(2, 3))
        a = np.arange(6).reshape(2, 3)
        b = np.arange(6).reshape(2, 3) + 6

        for axes in None, (1, 2, 0), (2, 1, 0):
            n = nodes.Stack(name='stack', num_inputs=2, axes=axes)
            n.input_1.connect(x2)
            n.input_0.connect(x1)
            n.transform()

            fx = nodes.compile(inputs=[x1, x2],
                               outputs=[n.output])

            z = fx(a, b)[0]
            expected = np.array([a, b])
            if axes:
                expected = np.transpose(expected, axes)
            np.testing.assert_equal(z, expected)

    def test_Dimshuffle(self):
        x1 = core.Input(name='x1', shape=(2, 3))
        a = np.zeros([2, 3])
        axes = [('x', 0, 1), (0, 1, 'x'), (1, 'x', 0)]
        shapes = [(1, 2, 3), (2, 3, 1), (3, 1, 2)]
        for ax, shp in zip(axes, shapes):
            n = nodes.Dimshuffle('dimshuffle', ax)
            n.input.connect(x1)
            n.transform()

            fx = nodes.compile(inputs=n.inputs.values(),
                               outputs=n.outputs.values())

            np.testing.assert_equal(fx(a)[0].shape, shp)

    def test_Slice(self):
        x1 = core.Input(name='x1', shape=(2, 3))
        a = np.arange(6).reshape(2, 3)
        slices = [(None, 1), (0, None), (1, 0)]
        ans = [a[:, 1], a[0, :], a[1, 0]]
        for slc, ans in zip(slices, ans):
            n = nodes.Slice('slice', slc)
            n.input.connect(x1)
            n.transform()

            fx = nodes.compile(inputs=n.inputs.values(),
                               outputs=n.outputs.values())

            np.testing.assert_equal(fx(a)[0], ans)

    def test_Log(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        log = nodes.Log('log')
        log.input.connect(x1)
        log.transform()

        fx = nodes.compile(inputs=log.inputs.values(),
                           outputs=log.outputs.values())

        a = np.array([[3, 1], [4, 7]], dtype=np.float32)
        z = fx(a)[0]
        np.testing.assert_almost_equal(z, np.log(a))

    def test_Multiply(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        a = np.array([[3, -1], [3, 7]])

        for w, shp in zip([-1, a], [None, a.shape]):
            n = nodes.Multiply(name='gain', weight_shape=shp)
            n.input.connect(x1)
            n.transform()

            fx = nodes.compile(inputs=n.inputs.values(),
                               outputs=n.outputs.values())

            np.testing.assert_equal(fx(a)[0], np.zeros_like(a))

            n.weight.value = w
            np.testing.assert_equal(fx(a)[0], w*a)

        n = nodes.Multiply(name='gain', weight_shape=(1, 2), broadcast=[0])
        n.input.connect(x1)
        n.transform()

        fx = nodes.compile(inputs=n.inputs.values(),
                           outputs=n.outputs.values())

        np.testing.assert_equal(fx(a)[0], np.zeros_like(a))

        n.weight.value = a[0].reshape(1, -1)
        np.testing.assert_equal(fx(a)[0], a*a[0].reshape(1, -1))

    def test_Max(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        a = np.array([[3, -1], [4, 7]])
        res = 7, np.array([4, 7]), np.array([3, 7])
        for idx, axis in enumerate([None, 0, 1]):
            n = nodes.Max('max', axis=axis)
            n.input.connect(x1)
            n.transform()

            fx = nodes.compile(inputs=n.inputs.values(),
                               outputs=n.outputs.values())

            np.testing.assert_equal(fx(a)[0], res[idx])

    def test_Min(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        a = np.array([[3, -1], [4, 7]])
        res = -1, np.array([3, -1]), np.array([-1, 4])
        for idx, axis in enumerate([None, 0, 1]):
            n = nodes.Min('min', axis=axis)
            n.input.connect(x1)
            n.transform()

            fx = nodes.compile(inputs=n.inputs.values(),
                               outputs=n.outputs.values())

            np.testing.assert_equal(fx(a)[0], res[idx])

    def test_Sum(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        a = np.array([[3, -1], [4, 7]])
        res = 13, np.array([7, 6]), np.array([2, 11])
        for idx, axis in enumerate([None, 0, 1]):
            n = nodes.Sum('sum', axis=axis)
            n.input.connect(x1)
            n.transform()

            fx = nodes.compile(inputs=n.inputs.values(),
                               outputs=n.outputs.values())

            np.testing.assert_equal(fx(a)[0], res[idx])

    def test_Mean(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        a = np.array([[3, -1], [4, 7]])
        res = 13 / 4.0, np.array([7, 6]) / 2.0, np.array([2, 11]) / 2.0
        for idx, axis in enumerate([None, 0, 1]):
            n = nodes.Mean('mean', axis=axis)
            n.input.connect(x1)
            n.transform()

            fx = nodes.compile(inputs=n.inputs.values(),
                               outputs=n.outputs.values())

            np.testing.assert_equal(fx(a)[0], res[idx])

    def test_NormalizeDim(self):
        x1 = core.Input(name='x1', shape=(1, 2, 3))
        a = np.array([[[3, 1, -1], [4, 0, 7]]], dtype=np.float32)
        expected = [np.sign(a),
                    a / np.sqrt(np.array([25, 1, 50])).reshape(1, 1, 3),
                    a / np.sqrt(np.array([11, 65])).reshape(1, 2, 1)]
        for axis, ans in enumerate(expected):
            n = nodes.NormalizeDim('l2norm', axis=axis, mode='l2')
            n.input.connect(x1)
            n.transform()

            fx = nodes.compile(inputs=n.inputs.values(),
                               outputs=n.outputs.values())
            np.testing.assert_almost_equal(fx(a)[0], ans)

    def test_SelectIndex(self):
        x1 = core.Input(name='x1', shape=(None, 2))
        idx = core.Input(name='idx', shape=(None,), dtype='int32')
        a = np.array([[3, -1], [4, 7]])
        i = np.array([1, 0])

        n = nodes.SelectIndex('select')
        n.input.connect(x1)
        n.index.connect(idx)
        n.transform()

        fx = nodes.compile(inputs=[x1, idx],
                           outputs=n.outputs.values())

        np.testing.assert_equal(fx(a, i)[0], np.array([-1, 4]))

    def test_SquaredEuclidean(self):
        a1 = np.array([[3, -1], [4, 7]])
        b1 = np.array([[1, -1], [4, 7]])
        a2 = np.array([3, -1])
        b2 = np.array([1, -1])

        z1 = np.power(a1 - b1, 2.0).sum(axis=1)
        z2 = np.power(a2 - b2, 2.0).sum()
        for a, b, z in zip([a1, a2], [b1, b2], [z1, z2]):
            x1 = core.Input(name='x1', shape=a.shape)
            x2 = core.Input(name='x2', shape=b.shape)
            n = nodes.SquaredEuclidean('sqeuclid')
            n.input_a.connect(x1)
            n.input_b.connect(x2)
            n.transform()

            fx = nodes.compile(inputs=[x1, x2],
                               outputs=n.outputs.values())
            np.testing.assert_equal(fx(a, b)[0], z)

    def test_Product(self):
        a1 = np.array([[3, -1], [4, 7]])
        b1 = np.array([[1, -1], [4, 7]])
        a2 = np.array([3, -1])
        b2 = np.array([1, -1])

        for a, b in zip([a1, a2], [b1, b2]):
            x1 = core.Input(name='x1', shape=a.shape)
            x2 = core.Input(name='x2', shape=b.shape)
            n = nodes.Product('product')
            n.input_a.connect(x1)
            with self.assertRaises(AssertionError):
                n.transform()
            n.input_b.connect(x2)
            self.assertTrue(n.is_ready())
            n.transform()

            fx = nodes.compile(inputs=[x1, x2],
                               outputs=n.outputs.values())
            np.testing.assert_equal(fx(a, b)[0], a*b)

    def test_Affine_linear(self):
        x1 = core.Input(name='x1', shape=(None, 2))
        a = np.array([[3, -1], [4, 7]])
        w = np.array([[1, -1], [2, -2], [3, -3]]).T
        b = np.ones(3)

        n = nodes.Affine(
            name='affine',
            input_shape=(None, 2),
            output_shape=(None, 3),
            act_type='linear')
        n.weights.value = w
        n.bias.value = b

        n.input.connect(x1)
        n.transform()

        fx = nodes.compile(inputs=[x1], outputs=n.outputs.values())
        np.testing.assert_equal(fx(a)[0], np.dot(a, w) + b)

    def test_Affine_relu(self):
        x1 = core.Input(name='x1', shape=(None, 2))
        a = np.array([[3, -1], [4, 7]])
        w = np.array([[1, -1], [2, -2], [3, -3]]).T
        b = np.ones(3)

        n = nodes.Affine(
            name='affine',
            input_shape=(None, 2),
            output_shape=(None, 3),
            act_type='relu')
        n.weights.value = w
        n.bias.value = b

        n.input.connect(x1)
        n.transform()

        fx = nodes.compile(inputs=[x1], outputs=n.outputs.values())
        np.testing.assert_equal(fx(a)[0], __relu__(np.dot(a, w) + b))

    def test_Affine_dropout(self):
        x1 = core.Input(name='x1', shape=(None, 2))
        dropout = core.Input(name='dropout', shape=None)
        a = np.array([[3, -1], [4, 7]])
        w = np.array([[1, -1], [2, -2], [3, -3]]).T
        b = np.ones(3)

        n = nodes.Affine(
            name='affine',
            input_shape=(None, 2),
            output_shape=(None, 3),
            act_type='linear')
        n.weights.value = w
        n.bias.value = b
        n.enable_dropout()

        n.input.connect(x1)
        with self.assertRaises(AssertionError):
            n.transform()
        n.dropout.connect(dropout)
        n.transform()

        fx = nodes.compile(inputs=[x1, dropout], outputs=n.outputs.values())

        np.testing.assert_equal(fx(a, 0.0)[0], np.dot(a, w) + b)
        self.assertGreaterEqual(np.equal(fx(a, 0.9)[0], 0.0).sum(), 1)

    def test_Affine_share_params(self):
        x = core.Input(name='x1', shape=(None, 2))
        a = np.array([[3, -1], [4, 7]])

        w = np.array([[1, -1], [2, -2], [3, -3]]).T
        b = np.ones(3)

        n1 = nodes.Affine(
            name='affine',
            input_shape=(None, 2),
            output_shape=(None, 3),
            act_type='linear')

        n2 = nodes.Affine(
            name='affine_copy',
            input_shape=(None, 2),
            output_shape=(None, 3),
            act_type='linear')

        n2.share_params(n1)

        n1.weights.value = w
        n1.bias.value = b

        np.testing.assert_equal(n1.weights.value, n2.weights.value)
        np.testing.assert_equal(n1.bias.value, n2.bias.value)

        n2.input.connect(x)
        n2.transform()

        fx = nodes.compile(inputs=[x], outputs=n2.outputs.values())
        np.testing.assert_equal(fx(a)[0], np.dot(a, w) + b)

        n1.weights.value *= 2
        np.testing.assert_equal(fx(a)[0], np.dot(a, 2*w) + b)

    def test_Conv3D_linear(self):
        x1 = core.Input(name='x1', shape=(None, 1, 2, 3))
        a = np.array([[3, -1], [4, 7], [2, -6]]).reshape(2, 3)
        w = np.array([[[1], [-2]],
                     [[-3], [4]],
                     [[5], [-6]]]).reshape(3, 2, 1)
        b = np.arange(3)

        # Note that convolutions flip the kernels
        z = np.array([[(a*wi[::-1]).sum(axis=0) + bi
                      for wi, bi in zip(w, b)]])

        n = nodes.Conv3D(
            name='conv3d',
            input_shape=(None, 1, 2, 3),
            weight_shape=(3, 1, 2, 1),
            act_type='linear')

        n.weights.value = w.reshape(3, 1, 2, 1)
        n.bias.value = b

        n.input.connect(x1)
        n.transform()

        fx = nodes.compile(inputs=[x1], outputs=n.outputs.values())
        np.testing.assert_equal(fx(a.reshape(1, 1, 2, 3))[0],
                                z.reshape(1, 3, 1, 3))

    def test_Conv3D_relu(self):
        x1 = core.Input(name='x1', shape=(None, 1, 2, 3))
        a = np.array([[3, -1], [4, 7], [2, -6]]).reshape(2, 3)
        w = np.array([[[1], [-2]],
                     [[-3], [4]],
                     [[5], [-6]]]).reshape(3, 2, 1)
        b = np.arange(3)

        # Note that convolutions flip the kernels
        z = np.array([[(a*wi[::-1]).sum(axis=0) + bi
                      for wi, bi in zip(w, b)]])

        # Reshape from convenience
        a = a.reshape(1, 1, 2, 3)
        z = z.reshape(1, 3, 1, 3)

        n = nodes.Conv3D(
            name='conv3d',
            input_shape=(None, 1, 2, 3),
            weight_shape=(3, 1, 2, 1),
            act_type='relu')

        n.weights.value = w.reshape(3, 1, 2, 1)
        n.bias.value = b

        n.input.connect(x1)
        n.transform()

        fx = nodes.compile(inputs=[x1], outputs=n.outputs.values())
        np.testing.assert_equal(fx(a)[0], __relu__(z))

    def test_Conv3D_dropout(self):
        x1 = core.Input(name='x1', shape=(None, 1, 2, 3))
        dropout = core.Input(name='dropout', shape=None)
        a = np.array([[3, -1], [4, 7], [2, -6]]).reshape(2, 3)
        w = np.array([[[1], [-2]],
                     [[-3], [4]],
                     [[5], [-6]]]).reshape(3, 2, 1)
        b = np.arange(3)

        # Note that convolutions flip the kernels
        z = np.array([[(a*wi[::-1]).sum(axis=0) + bi
                      for wi, bi in zip(w, b)]])

        # Reshape from convenience
        a = a.reshape(1, 1, 2, 3)
        z = z.reshape(1, 3, 1, 3)

        n = nodes.Conv3D(
            name='conv3d',
            input_shape=(None, 1, 2, 3),
            weight_shape=(3, 1, 2, 1),
            act_type='linear')

        n.enable_dropout()
        n.weights.value = w.reshape(3, 1, 2, 1)
        n.bias.value = b

        n.input.connect(x1)
        with self.assertRaises(AssertionError):
            n.transform()
        n.dropout.connect(dropout)
        n.transform()

        fx = nodes.compile(inputs=[x1, dropout], outputs=n.outputs.values())

        np.testing.assert_equal(fx(a, 0.0)[0], z)
        self.assertGreaterEqual(np.equal(fx(a, 0.9)[0], 0.0).sum(), 1)

    def test_RadialBasis(self):
        x = core.Input(name='x', shape=(None, 2))
        a = np.array([[3, -1], [4, 7]])
        w = np.array([[1, -1], [2, -2], [3, -3]]).T

        n = nodes.RadialBasis(
            name='radial',
            input_shape=x.shape,
            output_shape=(None, 3))
        n.weights.value = w.reshape(1, 2, 3)
        n.input.connect(x)
        n.transform()

        fx = nodes.compile(inputs=[x], outputs=n.outputs.values())
        z = np.power(a.reshape(2, 2, 1) - w.reshape(1, 2, 3),
                     2.0).sum(axis=1)
        np.testing.assert_equal(fx(a)[0], z)

if __name__ == "__main__":
    unittest.main()
