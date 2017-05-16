# -*- coding: utf-8 -*-
#
# helpers.py: Includes a set of rudimentary helper functions for the
# fourth assignment in TDT4173.
#
import os

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Could not import the matplotlib library. Please '
                      'refer to http://matplotlib.org/ for installation '
                      'instructions.')


def load_task1_data(train_path, test_path):
    """Load and return the classification dataset from assignment 1.

    Can work for other CSV files, but assumes that there is only one
    target value.

    Parameters
    ----------
    train_path : str
    test_path : str
    """
    # Load training data
    train = np.genfromtxt(train_path, delimiter=',', dtype=np.float32)
    X_train = train[:, :train.shape[1]-1]
    y_train = train[:, train.shape[1]-1:]
    # Load test data
    test = np.genfromtxt(test_path, delimiter=',', dtype=np.float32)
    X_test = test[:, :test.shape[1]-1]
    y_test = test[:, test.shape[1]-1:]

    return X_train, y_train, X_test, y_test


def plot_curves(*args):
    """Plot the input as curve plots using matplotlib.

    Make sure to run `plt.show()` with matplotlib imported as
    `import matplotlib.pyplot as plt` after you have used this function.

    The input argument to this function must be alternating keys and values:
    `plot_curves('label1', numpy.ndarray, 'label2', numpy.ndarray, ...)`

    Parameter
    ---------
    *args: see above
    """
    # Convert to `{key1: value1, key2: value2, ...}` format
    data = dict((a, b) for a, b in zip(args[0::2], args[1::2]))

    # Plot every input as a curve
    plt.figure()
    for key in data:
        plt.plot(range(data[key].shape[0]), data[key], label=key)
    plt.legend()


def load_mnist_tf(path='./mnist'):
    """Download and return the MNIST dataset using TensorFlow.

    Parameter
    ---------
    path : str
        The location of where you would like the download MNIST.
    """
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets(path, one_hot=True)


def create_dir(path):
    """Ensure that the input path points to an existing directory.

    Parameter
    ---------
    path : str
    """
    if not os.path.exists(path):
        os.makedirs(path)


def plot_samples(samples):
    """Return a plot of the input samples in a grid.

    The number of samples and sample size must be divisible by 2.

    Parameter
    ---------
    samples : numpy.ndarray
        Must have exactly two dimensions. The first is the sample
        number, while the second is the sample itself (a vector of
        numbers).
    """
    assert samples.shape[0] % 2 == 0,\
        ('Number of samples is not divisible by 2.')
    assert len(samples[0]) % 2 == 0,\
        ('Sample size is not divisible by 2.')

    grid_size = int(np.sqrt(samples.shape[0]))
    img_size = int(np.sqrt(len(samples[0])))

    figure = plt.figure(figsize=(grid_size, grid_size))
    grid = matplotlib.gridspec.GridSpec(grid_size, grid_size)
    grid.update(hspace=0.1, wspace=0.1)

    for idx, sample in enumerate(samples):
        ax = plt.subplot(grid[idx])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(sample.reshape(img_size, img_size), cmap=plt.cm.gray)

    return figure