optimus
=======
A python package for describing and serializing feed-forward (acyclic) signal processing graphs by decoupling topology and parameter from implementation in JSON. Currently, the primary purpose of optimus is to extend the functionality of Theano making it easier to build and reconstruct arbitrary neural networks in a language-agnostic manner.


Documentation
-------------
Unfortunately, documentation is rather sparse at the moment. Please refer to the demos found in `examples/` and the notebooks below in the interim.


Demonstration notebook
----------------------
What does optimus bring to Theano?  Here is a quick demonstration:

* Introduction notebook (soon): a brief introduction to building, saving, and re-loading a network.
* Basic Neural Networks (soon): using built-in data from scikit-learn, train a simple neural network classifier.
* [MNIST with a ConvNet](http://nbviewer.ipython.org/github/ejhumphrey/optimus/blob/master/examples/mnist.ipynb): demonstration using optimus to build a ConvNet for the MNIST dataset, showing the basics of building a graph and training it with real data.
https://github.com/

Installation
------------

While optimus is, and will for the foreseeable future be, in hyper-alpha state, the most reliable version known to date lives in the master branch of this repository.

To build and install from source, clone the repository locally and run the following:

```
git clone https://github.com/ejhumphrey/optimus.git
cd optimus
python setup.py build
python setup.py install
```

Depending on your system configuration, you may need administrator / sudo privileges to install.

That said, if you've made it this far, the mnist demo script, provided at`examples/mnist.py` should run without a hitch.

Alternatively, download or clone the repository and use `easy_install` to handle dependencies:

```
unzip optimus.zip
easy_install optimus
```
or
```
git clone https://github.com/ejhumphrey/optimus.git
easy_install optimus
```

Citing
------
...working on it...
