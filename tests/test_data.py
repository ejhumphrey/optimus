"""
"""

import unittest
import numpy as np
import optimus.data as D


class DataTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_entity(self):
        entity = D.Entity(a=3, b='im_a_string', c=[1, 2, 3], d=np.arange(5))

        self.assertEqual(
            entity.a.value,
            entity['a'].value,
            "Failed to initialize keys / attributes.")

        self.assertEqual(
            entity.a.value,
            3,
            "Failed to initialize an int.")

        self.assertEqual(
            entity.b.value.tostring(),
            'im_a_string',
            "Failed to initialize a string.")

        self.assertEqual(
            entity.c.value.tolist(),
            [1, 2, 3],
            "Failed to initialize a list.")

        self.assertEqual(
            entity.d.value.tolist(),
            range(5),
            "Failed to initialize a numpy array.")


if __name__ == "__main__":
    unittest.main()
