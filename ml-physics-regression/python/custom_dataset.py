""" Comprises 3.2K training examples and 3.2K test examples
    of the CMS electromagnetic calorimeter formatted
    as 28 * 28 = 784 pixel monochromatic images.
    The images are labeled with the normalized truth energy.
"""
from collections import namedtuple

import numpy as np
import os

DATADIR = '/home/mwaqas/work/ml-physics/data'

IMAGES = 'eplus_Ele-Eta0PhiPiOver2-Energy20to100_V2.npy'
LABELS = 'eplus_Ele-Eta0PhiPiOver2-Energy20to100_V2.txt'

ENERGY_NORM_FACTOR = 100.


def read_data(threshold):
    """ Read numpy arrays.
        Select images with energy above the threshold.
        Return images and normalized labels.
    """
    images = np.load(os.path.join(DATADIR, IMAGES))
    labels = np.loadtxt(os.path.join(DATADIR, LABELS), np.float32)
    p = [i for i, img in enumerate(images) if np.sum(img) > threshold]

    return images[p]/ENERGY_NORM_FACTOR, labels[p]/ENERGY_NORM_FACTOR


def load_dataset(threshold):
    """ Split into training and validation sets.
        Return namedtuple.
    """
    X, y = read_data(threshold)
    X = np.reshape(X, [-1, 28*28])
    m = min(len(y), 6400)

    X_train, X_val = X[:m//2], X[m//2:m]
    y_train, y_val = y[:m//2], y[m//2:m]

    Samples = namedtuple('Samples', 'images labels')
    Dataset = namedtuple('Dataset', 'train validation')
    return Dataset(Samples(X_train, y_train), Samples(X_val, y_val))


if __name__ == '__main__':
    dataset = load_dataset(threshold=10.)
    print(dataset.train.images.max())
