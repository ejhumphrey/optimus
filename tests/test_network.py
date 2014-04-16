"""
"""

import unittest

import optimus.network as N
from optimus.network.core import Port


class NetworkTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_connection_manager(self):
        connection_map = {'a': ['b'],
                          'b': ['a', 'c']}
        a = Port('a')
        b = Port('b')
        c = Port('c')

        edges = [(a, b), (b, a), (b, c)]
        connection_manager = N.ConnectionManager(edges)
        self.assertEqual(
            connection_manager.connection_map,
            connection_map,
            "Failed to parse edges properly.")

        cm_string = N.json.dumps(connection_manager)
        cm_reloaded = N.json.loads(cm_string)
        self.assertEqual(
            cm_reloaded.connection_map,
            connection_map,
            "Failed to de/serialize properly.")


if __name__ == "__main__":
    unittest.main()
