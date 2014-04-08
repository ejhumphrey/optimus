
from . import core
# from . import json
import numpy as np
from theano.tensor import grad
from collections import OrderedDict
from theano import function


def named_list(items):
    """TODO(ejhumphrey): write me."""
    return OrderedDict([(obj.name, obj) for obj in items])


class ConnectionManager(core.JObject):
    """TODO(ejhumphrey): write me."""
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

    Property attributes are named dictionaries, while the corresponding private
    variables are lists.
    """
    TOTAL_LOSS = 'total_loss'

    def __init__(self, name, inputs, nodes, edges, outputs,
                 losses=None, update_param=None, constraints=None,
                 chunk_size=250):
        """writeme."""

        self._inputs = inputs
        self._nodes = nodes
        self._connection_map = ConnectionManager(edges).connection_map
        self._outputs = outputs
        if losses is None:
            losses = list()
        self._losses = losses

        self.loss = core.Port(name=self.TOTAL_LOSS)
        self.loss.variable = 0.0
        if self.TOTAL_LOSS in self._outputs and not self._losses:
            raise ValueError(
                "At least one Loss must be provided to 'losses' if the graph "
                "is to compute '%s' as an output." % self.TOTAL_LOSS)

        self.chunk_size = chunk_size
        # Disable chunking for trainers.
        if update_param:
            self.chunk_size = None
        self.update_param = update_param

        if constraints is None:
            constraints = list()
        self.constraints = constraints

        self.ports = OrderedDict()
        self.params = OrderedDict()
        self.updates = OrderedDict()
        self.outputs = OrderedDict()
        for port in outputs:
            if port == self.TOTAL_LOSS:
                self.outputs[self.TOTAL_LOSS] = None
            else:
                self.outputs[port.name] = None

        self._fx = None
        self.__wire__()

    @property
    def __json__(self):
        return dict(
            inputs=self._inputs,
            nodes=self._nodes,
            connection_map=self._connection_map,
            outputs=self._outputs,
            losses=self._losses,
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

    def __wire__(self):
        """Walk inputs to outputs via connection map, collecting
         x  Active inputs
         x  Params touched
         x  Active outputs

        Then, after populating the necessary objects, compile the corresponding
        theano function.

        TODO(ejhumphrey): This method is rather big. Perhaps break it down into
        smaller pieces?
        """
        # Collect all sources that are inputs.
        input_ports = dict()
        for source_name in self._connection_map:
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

        local_map = self._connection_map.copy()
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

        # Sum all active losses into the total loss.
        for loss in self._losses:
            if loss.cost.name in self.ports:
                self.loss.variable += loss.cost.variable
        self.ports[self.TOTAL_LOSS] = self.loss

        # Map configured ports to the requested outputs.
        for port_name in self.outputs:
            if port_name in self.ports:
                self.outputs[port_name] = self.ports[port_name]

        # Make sure that the loss is actually set when requested as an output.
        if not self.outputs.get(self.TOTAL_LOSS, True):
            raise ValueError(
                "Requesting '%s' as an output, but the value has "
                "not been initialized." % self.TOTAL_LOSS)

        # Define SGD update rules
        # TODO(ejhumphrey): In the future, it would be super useful to at least
        #   make it optional to use different update params for different node
        #   parameters.
        if self.loss.variable and self.update_param:
            for param in self.params.values():
                gparam = grad(self.loss.variable, param.variable)
                gparam *= self.update_param.variable
                self.updates[param.variable] = param.variable - gparam

        self._fx = function(
            inputs=[x.variable for x in self._inputs],
            outputs=[x.variable for x in self.outputs.values()],
            updates=self.updates,
            allow_input_downcast=True)

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
        return self._fx(*args, **kwargs)


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
