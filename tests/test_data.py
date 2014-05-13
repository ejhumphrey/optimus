"""
"""

import unittest
import tempfile as tmp
import os
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

        key = 'my_key'
        fpath = tmp.mktemp(suffix=".hdf5", dir=tmp.gettempdir())
        fh = D.File(fpath)
        fh.add(key, entity)
        fh.close()

        fh = D.File(fpath)
        entity2 = fh.get(key)

        self.assertEqual(
            entity.a.value,
            entity2.a.value,
            "Could not reconstitute entity.a")

        self.assertEqual(
            entity.b.value.tostring(),
            entity2.b.value.tostring(),
            "Could not reconstitute entity.b")

        self.assertEqual(
            entity.c.value.tolist(),
            entity2.c.value.tolist(),
            "Could not reconstitute entity.c")

        self.assertEqual(
            entity.d.value.tolist(),
            entity2.d.value.tolist(),
            "Could not reconstitute entity.d")

        entity2.a.value = 4
        self.assertRaises(ValueError, fh.add, key, entity2)

        fh.add(key, entity2, True)
        fh.close()

        fh = D.File(fpath)
        entity3 = fh.get(key)
        self.assertEqual(len(fh), 1)
        self.assertEqual(entity3.a.value, 4)

if __name__ == "__main__":
    unittest.main()
