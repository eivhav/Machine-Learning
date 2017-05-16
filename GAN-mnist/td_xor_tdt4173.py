#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# tf_xor_tdt4173.py: TensorFlow example for learning the XOR function.
#
import sys
import helpers as helpers
import os

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

print(os.getcwd())

def plot_errors(error_lists):
    plt.plot([x[0] for x in error_lists[0]], [x[1] for x in error_lists[0]], 'r')
    plt.plot([x[0] for x in error_lists[1]], [x[1] for x in error_lists[1]], 'b')
    plt.axis([0, 1000, 0.0, 0.8])
    plt.show()


path = os.getcwd() + '/data/'
x_train, y_train, x_test, y_test = helpers.load_task1_data(path+'cl-train.csv', path+'cl-test.csv')

# Define dataset

# Define model entry-points
X = tf.placeholder(tf.float32, shape=(None, 2))
y = tf.placeholder(tf.float32, shape=(None, 1))

# Create weights and gather them in a dictionary

hidden_layer_size = [5, 5, 1]

weights = {'w1': tf.Variable(tf.random_uniform([2, hidden_layer_size[0]], -1, 1)),
           'b1': tf.Variable(tf.zeros([hidden_layer_size[0]])),
           'w2': tf.Variable(tf.random_uniform([hidden_layer_size[0], hidden_layer_size[1]], -1, 1)),
           'b2': tf.Variable(tf.zeros([hidden_layer_size[1]])),
            'w3': tf.Variable(tf.random_uniform([hidden_layer_size[1], hidden_layer_size[2]], -1, 1)),
           'b3': tf.Variable(tf.zeros([1]))}

# Define model as a computational graph
z1 = tf.add(tf.matmul(X, weights['w1']), weights['b1'])
h1 = tf.sigmoid(z1)
z2 = tf.add(tf.matmul(h1, weights['w2']), weights['b2'])
h2 = tf.sigmoid(z2)
z3 = tf.add(tf.matmul(h2, weights['w3']), weights['b3'])
y_hat = tf.sigmoid(z3)

# Define error functions
error = - tf.reduce_mean(tf.multiply(y, tf.log(y_hat)) + tf.multiply(1 - y, tf.log(1 - y_hat)))

# Specify which optimiser to use (`lr` is the learning rate)
#lr = 0.01
#optimiser = tf.train.GradientDescentOptimizer(lr).minimize(error, var_list=weights.values())
optimiser = tf.train.AdamOptimizer(0.1).minimize(error, var_list=weights.values())

# Generate Op that initialises global variables in the graph
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initialise variables and start the session
    sess.run(init)

    # Run a set number of epochs
    nb_epochs = 1000

    errors = [[], []]

    for epoch in range(nb_epochs):
        sess.run(optimiser, feed_dict={X: x_train, y: y_train})

        # Print out some information every nth iteration

        train_err = sess.run(error, feed_dict={X: x_train, y: y_train})
        test_err = sess.run(error, feed_dict={X: x_test, y: y_test})
        errors[0].append([epoch, train_err])
        errors[1].append([epoch, test_err])

        if epoch % 100 == 0 or epoch == 999:
            print('Epoch: ', epoch, '\t train_error: ', train_err, '\t test_error: ', test_err)

    # Print out the final predictions
    predictions = sess.run(y_hat, feed_dict={X: x_test})
    print('\nFinal XOR function predictions:')
    for idx in range(len(predictions)):
        print('{} ~> {:.2f}'.format(x_test[idx], predictions[idx][0]), y_test[idx])

    plot_errors(errors)

del sess
sys.exit(0)