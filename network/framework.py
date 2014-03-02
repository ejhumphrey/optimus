
from . import core


def named_list(items):
    return dict([(obj.name, obj) for obj in items])


class Canvas(core.JObject):
    def __init__(self, inputs, nodes, losses, outputs):
        """
        inputs: list of Inputs
        nodes: list of Nodes
        """
        self._inputs = inputs
        self._nodes = nodes
        self._losses = losses
        self._outputs = outputs

    @property
    def __json__(self):
        return dict(
            inputs=self._inputs,
            nodes=self._nodes,
            losses=self._losses,
            outputs=self._outputs,
            type=self.type)

    @property
    def inputs(self):
        return named_list(self._inputs)

    @property
    def nodes(self):
        return named_list(self._nodes)

    @property
    def losses(self):
        return named_list(self._losses)

    @property
    def outputs(self):
        return named_list(self._outputs)


class ConnectionManager(core.JObject):
    def __init__(self, edges=None):
        """
        edges: list of Port tuples
        """
        if edges is None:
            edges = list()
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


class Graph(core.JObject):
    """writeme."""
    def __init__(self, name, canvas, outputs, edges,
                 loss=None, updates=None, constraints=None):
        """writeme.

        """
        self.canvas = canvas
        self.connection_map = ConnectionManager(edges).connection_map
        # Walk inputs to outputs via connection map
        input_ports = dict()
        for source_name in self.connection_map:
            source = self.canvas.inputs.get(source_name, None)
            if source:
                input_ports[source_name] = source

        for node in canvas.nodes.values():
            input_ports.update(node.inputs)
        for node in canvas.losses.values():
            input_ports.update(node.inputs)
        input_ports.update(canvas.outputs)

        local_map = self.connection_map.copy()
        modules = canvas.nodes.values() + canvas.losses.values()
        while local_map:
            nothing_happened = True
            for source_name in input_ports:
                print "Source: %s" % source_name
                if source_name in local_map:
                    sinks = local_map.pop(source_name)
                    print "Sinks: %s" % sinks
                    for sink_name in sinks:
                        nothing_happened = False
                        "Connecting %s to %s" % (source_name, sink_name)
                        input_ports[sink_name].connect(input_ports[source_name])
                    print "Remaining map: %s" % local_map
            for node in modules:
                print "Node: %s" % node.name
                if node.is_ready():
                    node.transform()
                    input_ports.update(node.outputs)
                    nothing_happened = False
                    break
            if nothing_happened:
                "Failsafe..."
                break
        self.ports = input_ports

    @property
    def __json__(self):
        return dict(inputs=self._inputs.values(), type=self.type)

    @classmethod
    def __json_init__(cls, inputs):
        inputs = [core.Input(**args) for args in inputs]
        return cls(inputs)

    def transform(self, manager):
        """writeme"""
        pass


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
