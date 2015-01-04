"""
"""

import unittest

import numpy as np
import theano
import theano.tensor as T
import optimus.network.functions as F


class FunctionTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_max_not_index(self):
        inputs = [T.matrix(), T.ivector()]
        fx = theano.function(inputs=inputs,
                             outputs=F.max_not_index(*inputs),
                             allow_input_downcast=True)
        x = np.array([[3, 4, 1], [3, 2, 1], [0, 7, 7]])
        idx = np.array([0, 0, 2])

        np.testing.assert_equal(fx(x, idx), np.array([4, 2, 7]))

    def test_min_not_index(self):
        inputs = [T.matrix(), T.ivector()]
        fx = theano.function(inputs=inputs,
                             outputs=F.min_not_index(*inputs),
                             allow_input_downcast=True)
        x = np.array([[3, 4, 1], [3, 2, 1], [10, 7, 7]])
        idx = np.array([0, 2, 1])

        np.testing.assert_equal(fx(x, idx), np.array([1, 2, 7]))

if __name__ == "__main__":
    unittest.main()
