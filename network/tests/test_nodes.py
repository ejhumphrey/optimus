"""
"""

import unittest

import numpy as np
import optimus.network.nodes as nodes
import optimus.network.core as core


class NodeTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_Node(self):
        pass

    def test_Accumulate(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        x2 = core.Input(name='x2', shape=(2, 2))

        acc = nodes.Accumulate(name='accumulate', num_inputs=2)
        acc.input_0.connect(x1)

        with self.assertRaises(AssertionError):
            acc.transform()

        acc.input_1.connect(x2)
        acc.transform()

        fx = nodes.compile(inputs=[x1, x2],
                           outputs=[acc.output])
        a = np.array([[3, -1], [3, 7]])
        b = np.array([[1, 2], [3, 4]])

        z = fx(a, b)[0]
        np.testing.assert_equal(z, np.array([[4, 1], [6, 11]]))

    def test_Concatenate(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        x2 = core.Input(name='x2', shape=(2, 2))
        a = np.array([[3, -1], [3, 7]])
        b = np.array([[1, 2], [3, 4]])

        for axis in range(2):
            cat = nodes.Concatenate('concatenate', axis=axis)
            cat.input_list.connect(x1)
            cat.input_list.connect(x2)
            cat.transform()

            fx = nodes.compile(inputs=[x1, x2],
                               outputs=[cat.output])

            z = fx(a, b)[0]
            np.testing.assert_equal(z, np.concatenate([a, b], axis=axis))

    def test_Stack(self):
        x1 = core.Input(name='x1', shape=(2, 3))
        x2 = core.Input(name='x2', shape=(2, 3))
        a = np.arange(6).reshape(2, 3)
        b = np.arange(6).reshape(2, 3) + 6

        for axes in None, (1, 2, 0), (2, 1, 0):
            n = nodes.Stack('stack', axes=axes)
            n.input_list.connect(x1)
            n.input_list.connect(x2)
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

    def test_Gain(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        gain = nodes.Gain('gain')
        gain.input.connect(x1)
        gain.transform()

        fx = nodes.compile(inputs=gain.inputs.values(),
                           outputs=gain.outputs.values())

        a = np.array([[3, -1], [3, 7]])
        np.testing.assert_equal(fx(a)[0], np.zeros_like(a))

        gain.weight.value = np.array(-1.0)
        np.testing.assert_equal(fx(a)[0], -1*a)

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
        x1 = core.Input(name='x1', shape=(None, 2))
        x2 = core.Input(name='x2', shape=(None, 2))

        a = np.array([[3, -1], [4, 7]])
        b = np.array([[1, -1], [4, 7]])

        n = nodes.SquaredEuclidean('sqeuclid')
        n.input_a.connect(x1)
        n.input_b.connect(x2)
        n.transform()

        fx = nodes.compile(inputs=[x1, x2],
                           outputs=n.outputs.values())
        np.testing.assert_equal(fx(a, b)[0], np.power(a - b, 2.0).sum(axis=1))

        x1 = core.Input(name='x1', shape=(None,))
        x2 = core.Input(name='x2', shape=(None,))

        a = np.array([3, -1])
        b = np.array([1, -1])

        n = nodes.SquaredEuclidean('sqeuclid')
        n.input_a.connect(x1)
        n.input_b.connect(x2)
        n.transform()

        fx = nodes.compile(inputs=[x1, x2],
                           outputs=n.outputs.values())
        np.testing.assert_equal(fx(a, b)[0], np.power(a - b, 2.0).sum())


if __name__ == "__main__":
    unittest.main()
