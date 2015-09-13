# optimus

A python package for describing and serializing feed-forward (acyclic) signal processing graphs by decoupling topology and parameter from implementation in JSON. Currently, the primary purpose of optimus is to extend the functionality of Theano making it easier to build and reconstruct arbitrary neural networks in a language-agnostic manner.


## Documentation

Unfortunately, documentation is rather sparse at the moment. Please refer to the demos found in `examples/` and the notebooks below in the interim.


## Demonstration notebook

What does optimus bring to Theano?  Here is a quick demonstration:

* Introduction notebook (soon): a brief introduction to building, saving, and re-loading a network.
* Basic Neural Networks (soon): using built-in data from scikit-learn, train a simple neural network classifier.
* [MNIST with a ConvNet](http://nbviewer.ipython.org/github/ejhumphrey/optimus/blob/master/examples/mnist.ipynb): demonstration using optimus to build a ConvNet for the MNIST dataset, showing the basics of building a graph and training it with real data.
https://github.com/


## Installation

The easiest way to install `optimus` is with pip:

```
$ pip install git+git://github.com/ejhumphrey/optimus.git
```

Alternatively, you can clone the repository and do it the hard way:

```
$ cd ~/to/a/good/place
$ git clone https://github.com/ejhumphrey/optimus.git
$ cd optimus
$ python setup.py build
$ [sudo] python setup.py install
```

## Testing your installation

Clone the repository and run the tests directly; `nose` is recommended, and installed as a dependency:

```
$ cd {wherever_you_cloned_it}/optimus
$ nosetests
```

If you've made it this far, the mnist demo script, provided at`examples/mnist.py` should run without a hitch.


Citing
------
...working on it...
