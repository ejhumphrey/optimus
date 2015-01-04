"""Graph-level tests.

TODO(ejhumphrey): Write more tests, guh!
"""

import unittest

import optimus


class NetworkTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_connection_manager(self):
        connection_map = {'a': ['b'],
                          'b': ['a', 'c']}
        a = optimus.Port('a')
        b = optimus.Port('b')
        c = optimus.Port('c')

        edges = [(a, b), (b, a), (b, c)]
        connection_manager = optimus.ConnectionManager(edges)
        self.assertEqual(
            connection_manager.connections,
            connection_map,
            "Failed to parse edges properly.")

if __name__ == "__main__":
    unittest.main()
