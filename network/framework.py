
from . import core


class Canvas(core.JObject):
    def __init__(self, inputs, nodes, losses, outputs):

        self.inputs = inputs
        self.nodes = nodes
        self.losses = losses
        self.outputs = outputs

    @property
    def __json__(self):
        return dict(
            inputs=self.inputs,
            nodes=self.nodes,
            losses=self.losses,
            outputs=self.outputs,
            type=self.type)

    def collect(self):
        pass


class ConnectionManager(core.JObject):
    def __init__(self, edges):
        """
        edges: list of Port tuples
        """
        self.connection_map = dict()
        for source, sink in edges:
            self.add_edge(source, sink)

    @property
    def __json__(self):
        return dict(connection_map=self.connection_map, type=self.type)

    @classmethod
    def __json_init__(cls, **kwargs):
        manager = cls()
        manager.connection_map = kwargs.get("connection_map")
        return manager

    def add_edge(self, source, sink):
        """
        source: Port
            Symbolic data source (from)
        sink: Port
            Symbolic data sink (to)
        """
        if not source.name in self.connection_map:
            self.connection_map[source.name] = list()
        self.connection_map[source.name].append(sink.name)


# class Graph(core.JObject):
#     """writeme."""
#     def __init__(self, inputs=None, connections=None, outputs=None,
#                  losses=None, updates=None, constraints=None):
#         """
#         All arguments are native Python types.

#         inputs: list of arg dictionaries
#         outputs: list of strings
#         """

#         self._outputs = dict()

#         if inputs is None:
#             inputs = list()
#         if connections is None:
#             connections = dict()
#         if outputs is None:
#             outputs = list()
#         if losses is None:
#             losses = dict()
#         if updates is None:
#             updates = dict()
#         if constraints is None:
#             constraints = dict()

#         self._inputs = dict([(obj.name, obj) for obj in inputs])

#         # Native data structure
#         # self.connections = Struct([(k, v) for k, v in connections.items()])
#         # self.outputs = Struct()

#         # Necessary? Could implicitly sum all losses to a reserved key.
#         # self.losses = None

#     @property
#     def __json__(self):
#         return dict(inputs=self._inputs.values(), type=self.type)

#     @classmethod
#     def __json_init__(cls, inputs):
#         inputs = [core.Input(**args) for args in inputs]
#         return cls(inputs)

#     def transform(self, manager):
#         """writeme"""
#         pass


# class Driver(Struct):
#     """writeme."""
#     def __init__(self, nodes=None, graphs=None):
#         """writeme."""
#         self.nodes = Struct()
#         if nodes is None:
#             nodes = dict()
#         if graphs is None:
#             graphs = dict()
#         for name,
#         [(k, NodeFactory(v)) for k, v in nodes.items()])
#         self.graphs = Struct([(k, Graph(**v)) for k, v in graphs.items()])
