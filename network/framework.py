
from . import core
from . import json
import numpy as np
import theano.tensor as T
from collections import OrderedDict


def named_list(items):
    """TODO(ejhumphrey): write me."""
    return dict([(obj.name, obj) for obj in items])


# class Canvas(core.JObject):
#     def __init__(self, inputs, nodes, losses, outputs):
#         """
#         inputs: list of Inputs
#         nodes: list of Nodes
#         """
#         self._inputs = inputs
#         self._nodes = nodes
#         self._losses = losses
#         self._outputs = outputs

#     @property
#     def __json__(self):
#         return dict(
#             inputs=self._inputs,
#             nodes=self._nodes,
#             losses=self._losses,
#             outputs=self._outputs,
#             type=self.type)

#     @property
#     def inputs(self):
#         return named_list(self._inputs)

#     @property
#     def nodes(self):
#         return named_list(self._nodes)

#     @property
#     def losses(self):
#         return named_list(self._losses)

#     @property
#     def outputs(self):
#         return named_list(self._outputs)


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
    """Graph Implementation

    Private variables are lists
    Property attributes are named dictionaries

    """
    def __init__(self, name, inputs, nodes, edges, outputs,
                 losses=None, update_param=None, constraints=None,
                 chunk_size=250):
        """writeme."""

        self._inputs = inputs
        self._nodes = nodes
        self.connection_map = ConnectionManager(edges).connection_map
        self._outputs = outputs
        if losses is None:
            losses = list()
        self._losses = losses

        self.chunk_size = chunk_size
        # Disable chunking for trainers.
        if update_param:
            self.chunk_size = None
        self.update_param = update_param

        if constraints is None:
            constraints = list()
        self.constraints = constraints

        self.ports = {}
        self.params = dict()
        # Declare outputs
        self.outputs = OrderedDict()
        for port in outputs:
            if port == 'loss':
                self.outputs['loss'] = None
            else:
                self.outputs[port.name] = None

        self._fx = None
        # self.__wire__()
        # self.__compile__()

    @property
    def __json__(self):
        return dict(
            inputs=self._inputs,
            nodes=self._nodes,
            outputs=self._outputs,
            type=self.type)

    @property
    def inputs(self):
        return named_list(self._inputs)

    @property
    def nodes(self):
        return named_list(self._nodes)

    # @property
    # def outputs(self):
    #     return named_list(self._outputs)

    @property
    def losses(self):
        return named_list(self._losses)

    def __wire__(self):
        # Need to reset the nodes before proceeding.

        # Walk inputs to outputs via connection map, collecting
        # x  Active inputs
        # x  Params touched
        #    Active outputs

        # Collect all sources that are inputs.
        input_ports = dict()
        for source_name in self.connection_map:
            source = self.inputs.get(source_name, None)
            if source:
                input_ports[source_name] = source

        # Collect the Inputs of all nodes...
        for node in self.nodes.values():
            input_ports.update(node.inputs)
        # ...losses...
        for node in self.losses.values():
            input_ports.update(node.inputs)
        # ...and outputs.
        # input_ports.update(self.outputs)

        local_map = self.connection_map.copy()
        # print "All Ports: \n%s" % json.dumps(input_ports, indent=2)
        # print "Local Connection Map: \n%s" % json.dumps(local_map, indent=2)
        modules = self.nodes.values() + self.losses.values()
        # This could be smarter, but it will certainly terminate.
        while local_map:
            nothing_happened = True
            for source_name in input_ports:
                if source_name in local_map:
                    sinks = local_map.pop(source_name)
                    for sink_name in sinks:
                        nothing_happened = False
                        print "Connecting %s -> %s" % (source_name, sink_name)
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

    # @property
    # def __json__(self):
    #     return dict(inputs=self._inputs.values(), type=self.type)

    @classmethod
    def __json_init__(cls, inputs):
        raise NotImplementedError("Haven't fixed this yet.")
        inputs = [core.Input(**args) for args in inputs]
        return cls(inputs)

    def __call__(self, *args, **kwargs):
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
