"""Demonstrating how to use Optimus with the MNIST dataset.
"""

from optimus.data import Entity
from optimus.data import Feature
import cPickle


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
