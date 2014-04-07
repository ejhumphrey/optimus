
from . import core
import numpy as np
import theano.tensor as T


def named_list(items):
    """TODO(ejhumphrey): write me."""
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
    def __init__(self, edges):
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
        manager = cls(None)
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
    def __init__(self, name, inputs, nodes, edges, outputs,
                 loss=None, update_param=None, constraints=None,
                 chunk_size=250):
        """writeme."""
        if loss is None:
            loss = 0

        self.chunk_size = chunk_size
        self.ports = {}
        self.inputs = dict()
        self.outputs = dict([(key, None) for key in outputs])
        self.params = dict()
        self.loss = 0
        self._fx = None
        self.connection_map = ConnectionManager(edges).connection_map
        self.__wire__()

        # Configure updates
        if update_param:
            self.chunk_size = None
        # Configure constraints
        self.__compile__()

    def __wire__(self):
        # Need to reset the nodes before proceeding.

        # Walk inputs to outputs via connection map, collecting
        # x  Active inputs
        # x  Params touched
        #    Active outputs
        input_ports = dict()
        for source_name in self.connection_map:
            source = self.canvas.inputs.get(source_name, None)
            if source:
                input_ports[source_name] = source
                self.inputs.update({source_name: source})

        for node in self.canvas.nodes.values():
            input_ports.update(node.inputs)
        for node in self.canvas.losses.values():
            input_ports.update(node.inputs)
        input_ports.update(self.canvas.outputs)

        local_map = self.connection_map.copy()
        modules = self.canvas.nodes.values() + self.canvas.losses.values()

        # This could be smarter, but it will certainly terminate.
        while local_map:
            nothing_happened = True
            for source_name in input_ports:
                if source_name in local_map:
                    sinks = local_map.pop(source_name)
                    for sink_name in sinks:
                        nothing_happened = False
                        print "Connecting %s to %s" % (source_name, sink_name)
                        input_ports[sink_name].connect(
                            input_ports[source_name])
            for node in modules:
                if node.is_ready():
                    print "Transforming %s" % node
                    node.transform()
                    self.params.update(node.params)
                    input_ports.update(node.params)
                    input_ports.update(node.outputs)
                    nothing_happened = False
            if nothing_happened:
                print "Your logic is poor, but we can help."
                print "\t%s" % local_map
                break
        self.ports = input_ports

    def __compile__(self):
        for key in self.outputs:
            if key in self.ports:
                self.outputs.update({key: self.ports[key]})
        # Special Case
        if "loss" in self.outputs:
            self.outputs['loss'] = self.loss
        # Now call theano.function?

    @property
    def __json__(self):
        return dict(inputs=self._inputs.values(), type=self.type)

    @classmethod
    def __json_init__(cls, inputs):
        raise NotImplementedError("Haven't fixed this yet.")
        inputs = [core.Input(**args) for args in inputs]
        return cls(inputs)

    def __call__(self, **kwargs):
        """writeme"""
        # Needs internal buffering strategy -- check if the value of each kwarg
        # is an np.ndarray? if not, map it over all chunks. This should be a
        # separate function though...
        # if self.chunk_size is None and len(kwargs.values()[0]) >
        return self._fx(**kwargs)


def data_stepper(chunk_size=250, **kwargs):
    """Generator to chunk unweildy inputs into smaller bites."""
    constants = dict()
    arrays = dict()
    for key, value in kwargs:
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            constants[key] = value

    if not arrays:
        yield constants
        raise StopIteration

    num_chunks = int(np.ceil(len(arrays.values()[0]) / float(chunk_size)))
    for n in range(num_chunks):
        i0, i1 = n*chunk_size, (n+1)*chunk_size
        array_chunk = dict([(key, value[i0:i1])
                            for key, value in arrays.iteritems()])
        array_chunk.update(constants)
        yield array_chunk
