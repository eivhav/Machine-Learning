#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# tf_wgan_tdt4173.py: Rudimentary implementation of a wasserstein GAN that
# is to be used as a base for the second programming task in assignment 4.
#
import sys

import numpy as np
import tensorflow as tf

import helpers

import matplotlib.pyplot as plt

def plot_errors(error_lists, nb_e):
    plt.plot([x[0] for x in error_lists[0]], [x[1] for x in error_lists[0]], 'r')
    plt.plot([x[0] for x in error_lists[1]], [x[1] for x in error_lists[1]], 'b')
    plt.axis([0, 5000, -1.5, 1.5])
    plt.show()


# You might want to alter the learning rate, number of epochs, and batch size
lr = 0.0005
nb_epochs = 20000
batch_size = 200

# Set to `None` if you do not want to write out images
path_to_images = './generated_images'

z_size = 10
x_size = 28*28


# Defined at the top because we need it for initialising weights
def create_weights(shape):
    # See paper by Xavier Glorot and Yoshua Bengio for more information:
    # "Understanding the difficulty of training deep feedforward neural networks"
    # We employ the Caffe version of the initialiser: 1/(in degree)
    return tf.random_normal(shape, stddev=1/shape[0])

#
# Creation of generator and discriminator networks START here
# Task (a) is to improve the generator and discriminator networks as they
# currently do not do very much
#

# Define weight matrices for the generator
# Note: Input of the first layer *must* be `z_size` and the output of the
# *last* layer must be `x_size`
weights_G = {'w1': tf.Variable(create_weights((z_size, x_size))),
             'b1': tf.Variable(tf.zeros(x_size)),
             'w2': tf.Variable(create_weights((x_size, x_size))),
             'b2': tf.Variable(tf.zeros(x_size))}

def generator(z, weights):
    h1 = tf.nn.relu(tf.add(tf.matmul(z, weights['w1']), weights['b1']))
    out = tf.nn.sigmoid(tf.add(tf.matmul(h1, weights['w2']), weights['b2']))

    # Return model and weight matrices
    return out


# Define weight matrices for the discriminator
# Note: Input will always be `x_size` and output will always be 1
weights_D = {'w1': tf.Variable(create_weights((x_size, x_size))),
             'w2': tf.Variable(create_weights((x_size, 1))),
             'b1': tf.Variable(tf.zeros(x_size)),
             'b2': tf.Variable(tf.zeros(1))
             }

def discriminator(x, weights):
    h1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), weights['b1']))
    out = tf.nn.sigmoid(tf.add(tf.matmul(h1, weights['w2']), weights['b2']))

    # Return model and weight matrices
    return out

#
# Creation of generator and discriminator networks END here
#

# Weight clipping (default `c` from the WGAN paper)
c = 0.01
clipped_D = [w.assign(tf.clip_by_value(w, -c, c)) for w in weights_D.values()]

# Definition of how Z samples are generated
z_sampler = lambda nb, dim: np.random.uniform(-1.0, 1.0, size=(nb, dim))

# Load MNIST
mnist = helpers.load_mnist_tf('./mnist')

# Define model entry-points (Z - generator, X - discriminator)
Z = tf.placeholder(tf.float32, shape=(None, z_size))
X = tf.placeholder(tf.float32, shape=(None, x_size))

# Define the different components of a GAN
sample = generator(Z, weights_G)
fake_hat = discriminator(sample, weights_D)
real_hat = discriminator(X, weights_D)

# Define error functions
error_G = - tf.reduce_mean(fake_hat)
error_D = tf.reduce_mean(real_hat) - tf.reduce_mean(fake_hat)

# Specify that we will use RMSProp (one optimiser for each model)
optimiser_G = tf.train.RMSPropOptimizer(lr).minimize(error_G, var_list=weights_G.values())
optimiser_D = tf.train.RMSPropOptimizer(lr).minimize(-error_D, var_list=weights_D.values())

# Generate Op that initialises global variables in the graph
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initialise variables and start the session
    sess.run(init)

    if path_to_images:
        helpers.create_dir(path_to_images)

    # Run a set number of epochs (default `n_critic` from the WGAN paper)
    n_critic = 5
    errors = [[], []]
    for epoch in range(nb_epochs):
        for critic in range(n_critic):
            # Retrieve a batch from MNIST
            X_batch, critic = mnist.train.next_batch(batch_size)

            # Clip weights and run one step of the optimiser for D
            sess.run(clipped_D)
            sess.run(optimiser_D, feed_dict={Z: z_sampler(batch_size, z_size), X: X_batch})

        # Run one step of the optimiser for G
        sess.run(optimiser_G, feed_dict={Z: z_sampler(batch_size, z_size)})

        err_G = sess.run(error_G, feed_dict={Z: z_sampler(batch_size, z_size)})
        err_D = sess.run(error_D, feed_dict={Z: z_sampler(batch_size, z_size), X: X_batch})

        errors[0].append([epoch, err_G])
        errors[1].append([epoch, err_D])

        # Print out some information every nth iteration
        if epoch % 20 == 0:

            print('Epoch: ', epoch, '\t Generator error:\t {:.4f}'.format(err_G),
                  '\t Discriminator error:\t {:.4f}'.format(err_D))



        # Plot the image generated from 64 different samples to a directory
        if path_to_images and epoch % 500 == 0:
            samples = sess.run(sample, feed_dict={Z: z_sampler(64, z_size)})

            figure = helpers.plot_samples(samples)
            plt.savefig('{}/{}.png'.format(path_to_images, str(epoch)), bbox_inches='tight')
            plt.close()

    plot_errors(errors, nb_epochs)

del sess
sys.exit(0)