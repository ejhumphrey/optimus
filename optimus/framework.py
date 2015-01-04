
import numpy as np
from theano.tensor import grad
from collections import OrderedDict
from theano import function
# from theano import shared
import time
import os

import json
from .core import JObject
# from .core import Port
# from optimus.util import concatenate_data


def named_list(items):
    """TODO(ejhumphrey): write me."""
    return OrderedDict([(obj.name, obj) for obj in items])


def random_init(param, mean=0.0, std=0.025):
    param.value = np.random.normal(mean, std, size=param.shape)


class ConnectionManager(object):
    """TODO(ejhumphrey): write me."""
    def __init__(self, edges):
        """
        edges: list of Port tuples
        """
        if edges is None:
            edges = list()
        self._connection_map = dict()
        for source, sink in edges:
            if not hasattr(source, 'variable'):
                raise ValueError("Invalid source: %s" % source)
            if not hasattr(sink, 'variable'):
                raise ValueError("Invalid sink: %s" % sink)
            self.add_edge(source, sink)

    def add_edge(self, source, sink):
        """
        source: Port
            Data source (from)
        sink: Port
            Data sink (to)
        """
        if not source.name in self._connection_map:
            self._connection_map[source.name] = list()
        self._connection_map[source.name].append(sink.name)
        if sink is None:
            raise ValueError(
                "Sink for source ('%s') does not exist! "
                "Did you forget to create it?" % source.name)

    @property
    def connections(self):
        return self._connection_map.copy()

    @property
    def edges(self):
        edges = []
        for source, sinks in self.connections.items():
            edges.extend([(source, sink) for sink in sinks])
        return edges


class Graph(JObject):
    """Graph Implementation

    Property attributes are named dictionaries, while the corresponding private
    variables are lists.
    """
    def __init__(self, name, inputs, nodes, connections, outputs,
                 loss=None, updates=None, constraints=None,
                 chunk_size=250, verbose=False, momentum=0):
        """writeme."""

        self.name = name
        self._inputs = inputs
        self._nodes = nodes
        self._connections = connections
        self._outputs = outputs
        self.verbose = verbose
        self._loss = loss

        self.chunk_size = chunk_size
        # Disable chunking for trainers.
        if updates:
            self.chunk_size = None
        self._updates = updates

        if constraints is None:
            constraints = list()
        self.constraints = constraints

        self.ports = OrderedDict()
        self.params = OrderedDict()
        self.updates = OrderedDict()
        self.outputs = OrderedDict()
        for port in outputs:
            self.outputs[port.name] = port

        self.momentum = momentum
        self._fx = None
        self.__wire__()

    @property
    def __json__(self):
        return dict(
            name=self.name,
            inputs=self._inputs,
            nodes=self._nodes,
            connections=self._connections,
            outputs=self._outputs,
            loss=self._loss,
            updates=self._updates,
            type=self.type)

    @property
    def inputs(self):
        return named_list(self._inputs)

    @property
    def nodes(self):
        return named_list(self._nodes)

    @property
    def param_values(self):
        return dict([(k, self.params[k].value) for k in self.params])

    @param_values.setter
    def param_values(self, values):
        for k in values.keys():
            if k in self.params:
                self.params[k].value = values[k]
            else:
                print "Received erroneous parameter: %s" % k

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
        for source_name in self._connections:
            source = self.inputs.get(source_name, None)
            if source:
                input_ports[source_name] = source

        # Collect the Inputs of all nodes...
        for node in self.nodes.values():
            node.reset()
            input_ports.update(node.inputs)

        for port in self.outputs.values():
            # TODO(ejhumphrey): Why is this check here?
            if not port is None:
                port.reset()
                input_ports[port.name] = port

        local_map = self._connections.copy()
        # if self.verbose:
        #     print "All Ports: \n%s" % json.dumps(input_ports, indent=2)
        #     print "Connection Map: \n%s" % json.dumps(local_map, indent=2)
        nodes = self.nodes.values()
        # This could be smarter, but it will certainly terminate.
        while local_map:
            nothing_happened = True
            for source_name in input_ports:
                if source_name in local_map:
                    sinks = local_map.pop(source_name)
                    for sink_name in sinks:
                        nothing_happened = False
                        if self.verbose:
                            print "Connecting %s -> %s" % (source_name,
                                                           sink_name)
                        source = input_ports.get(source_name, None)
                        if source is None:
                            raise ValueError(
                                "Requested port `%s` does not exist; "
                                "Did you forget to include its node? "
                                "" % source_name)
                        input_ports[sink_name].connect(source)
            for node in nodes:
                if node.is_ready():
                    if self.verbose:
                        print ">>> Transforming %s" % node
                    node.transform()
                    self.params.update(node.params)
                    input_ports.update(node.params)
                    input_ports.update(node.outputs)
                    nothing_happened = False
                    break
            if nothing_happened:
                print "Your logic is poor, but we can help."
                for k, v in local_map.iteritems():
                    print "\t{ %s: %s }" % (k, v)
                return
        self.ports = input_ports

        # Map configured ports to the requested outputs.
        for port_name in self.outputs:
            # TODO(ejhumphrey): assert port_name is a valid variable
            if port_name in self.ports:
                self.outputs[port_name] = self.ports[port_name]
            else:
                raise ValueError(
                    "Expected '%s' as an output, but was not created "
                    "by the graph." % port_name)

        # Define SGD update rules
        # TODO(ejhumphrey): Break this out into something more atomic?
        if self._loss and self._updates:
            self.loss = self.ports.get(self._loss.name, None)
            if self.loss is None:
                raise ValueError(
                    "Requested loss `%s` is not among the "
                    "computed ports: \n%s" % (self._loss, self.ports.keys()))
            for input_name, param_names in self._updates.iteritems():
                eta = self.inputs.get(input_name)
                for param_name in param_names:
                    param = self.params[param_name]
                    gparam = grad(self.loss.variable, param.variable)
                    gparam *= eta.variable
                    self.updates[param.variable] = param.variable - gparam

        self._fx = function(
            inputs=[x.variable for x in self._inputs],
            outputs=[x.variable for x in self.outputs.values()],
            updates=self.updates,
            allow_input_downcast=True)

    def __call__(self, *args, **kwargs):
        """writeme"""
        # Needs internal buffering strategy -- check if the value of each kwarg
        # is an np.ndarray? if not, map it over all chunks. This should be a
        # separate function though...
        # if self.chunk_size is None and len(kwargs.values()[0]) > ...?
        return OrderedDict([(k, v) for k, v in zip(self.outputs.keys(),
                                                   self._fx(*args, **kwargs))])

    def save_param_values(self, filename):
        """Serialize the graph's parameter values to disk.

        Parameters
        ----------
        filename: str
            Path on disk to save the parameter values.
        """
        np.savez(filename, **self.param_values)

    def load_param_values(self, filename):
        """Load a set of parameter values from disk.

        Parameters
        ----------
        filename: str
            Path on disk of parameter values to load.
        """
        self.param_values = np.load(filename)


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


class Driver(object):
    """
    """
    TIME_FMT = "%04d-%02d-%02d_%02dh%02dm%02ds"

    def __init__(self, graph, name='trainer',
                 output_directory=None,
                 log_file='training_stats.json',
                 init_param_file=None):
        self.graph = graph
        self.log_file = log_file
        self.output_directory = output_directory
        self.name = name

        if init_param_file:
            self.graph.load_param_values(init_param_file)

        if not output_directory is None:
            self.output_directory = output_directory
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)
            def_file = os.path.join(self.output_directory,
                                    "%s.json" % self.name)
            save(self.graph, def_file)

        self._stats = dict()
        self._last_params = None

    def fit(self, source, hyperparams, max_iter=10000, save_freq=250,
            print_freq=50, nan_exceptions=0):
        """Fit the internal graph to the given data source.

        Parameters
        ----------
        source : generator
            Yields dictionaries of data matching the inputs of the graph.
        hyperparams : dict
            Static hyperparameters for the graph; merged with the dynamic
            source.
        max_iter : int
            Maximum number of interations to run.
        save_freq : int
            Number of iterations between checkpoints.
        print_freq : int
            Number of iterations between displaying progress.
        nan_exceptions : int, default=0
            Number of NaNs to catch before stopping.
        """
        assert self.graph.loss, "Loss not set!"
        self._stats.update(
            checkpoints=[],
            start_time=time.asctime(),
            hyperparams=hyperparams,
            max_iter=max_iter,
            save_freq=save_freq)

        if save_freq is None or save_freq <= 0:
            save_freq = np.inf

        param_file_fmt = "%%s-%%0%dd-%s.npz" % (np.ceil(np.log10(max_iter)+1),
                                                self.TIME_FMT)
        self._last_params = self.graph.param_values
        try:
            for n_iter, data in enumerate(source):
                data.update(**hyperparams)
                outputs = self.graph(**data)
                self.update_stats(n_iter, outputs[self.graph.loss.name])
                if n_iter > 0 and (n_iter % save_freq) == 0:
                    if not self.output_directory is None:
                        self._save_params(n_iter, param_file_fmt)
                if (n_iter % print_freq) == 0:
                    self.print_last_stats()
                if n_iter >= max_iter:
                    break
                if not np.isfinite(outputs[self.graph.loss.name]):
                    print "Caught a non-finite loss at iteration: %d " % n_iter
                    if nan_exceptions <= 0:
                        print "Stopping."
                        break
                    print "Reseting parameter values and moving on..."
                    self.graph.param_values = self._last_params
                    nan_exceptions = nan_exceptions - 1
                self._last_params = self.graph.param_values

        except KeyboardInterrupt:
            print "Stopping early after %d iterations" % n_iter

    def update_stats(self, n_iter, loss):
        cpoint = dict(timestamp=time.asctime(),
                      n_iter=n_iter,
                      loss=float(loss))
        self._stats['checkpoints'].append(cpoint)

    def print_last_stats(self):
        cpoint = self._stats['checkpoints'][-1]
        print "[%s] %d / %d: %0.4f" % (cpoint['timestamp'],
                                       cpoint['n_iter'],
                                       self._stats['max_iter'],
                                       cpoint['loss'])

    def _save_params(self, n_iter, param_file_fmt):
        args = tuple([self.name, n_iter] + list(time.localtime()[:6]))
        param_file = os.path.join(self.output_directory, param_file_fmt % args)
        self.graph.save_param_values(param_file)

    def __del__(self):
        if not self.log_file or not self.output_directory:
            return
        log_file = os.path.join(self.output_directory, self.log_file)
        with open(log_file, 'w') as fp:
            json.dump(self._stats, fp, indent=2)