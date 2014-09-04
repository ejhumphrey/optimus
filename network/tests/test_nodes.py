"""
"""

import unittest

import numpy as np
import theano
import theano.tensor as T
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
        x1 = core.Input(name='x1', shape=(2,2))
        x2 = core.Input(name='x2', shape=(2,2))

        acc = nodes.Accumulate('accumulate')
        acc.input_list.connect(x1)
        acc.input_list.connect(x2)

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

    def test_Log(self):
        x1 = core.Input(name='x1', shape=(2, 2))
        log = nodes.Log('log')
        log.input.connect(x1)
        log.transform()

        fx = nodes.compile(inputs=log.inputs.values(),
                           outputs=log.outputs.values())

        a = np.array([[3, -1], [3, 7]])
        z = fx(a)[0]
        np.testing.assert_equal(z, np.log(a))

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


if __name__ == "__main__":
    unittest.main()
