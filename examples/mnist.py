"""Demonstrating how to use Optimus with the MNIST dataset.
"""

import cPickle
from matplotlib.pyplot import figure
import numpy as np
from optimus.data import Entity
from optimus.data import Feature


class Digit(Entity):
    """Define the features for an handwritten digit."""
    def __init__(self, image, label):
        """writeme"""
        Entity.__init__(self,
                        image=Feature(value=image),
                        label=Feature(value=label))


class InputDigit(Entity):
    """Define the features for an handwritten digit."""
    def __init__(self, name, image, label):
        """writeme"""
        Entity.__init__(self,
                        name=name,
                        image=Feature(value=image),
                        label=Feature(value=label))


def load_mnist(filename):
    """Load the MNIST dataset into Digit dictionaries.

    Parameters
    ----------
    filename : str
        Path to MNIST pickle file.

    Returns
    -------
    train, valid, test: dicts of Digits
        Digits are keyed by string integers in the order they were added.
    """
    dsets = []
    for dpoints in cPickle.load(open(filename)):
        dataset = dict()
        for image, label in zip(dpoints[0], dpoints[1]):
            key = "%05d" % len(dataset)
            dataset[key] = Digit(image.reshape(1, 28, 28), label)
        dsets.append(dataset)

    return dsets


def draw_image_posterior(image, posterior):
    """Draw an image and its corresponding posterior"""
    fig = figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(image.reshape(28, 28), interpolation='nearest', aspect='equal')
    ax2.bar(range(10), posterior, width=0.95)
    ax2.set_xticks(np.arange(10) + 0.5)
    ax2.set_xticklabels(range(10))
