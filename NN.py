import math
import tensorflow as tf
import numpy as np


def weight(name, shape, init='he'):
    assert init == 'he' and len(shape) == 2
    std = math.sqrt(2.0 / shape[0])
    initializer = tf.random_normal_initializer(stddev=std)

    var = tf.get_variable(name, shape, initializer=initializer)
    tf.add_to_collection('l2', tf.nn.l2_loss(var))
    return var


def bias(name, dim, initial_value=1e-2):
    return tf.get_variable(name, dim, initializer=tf.constant_initializer(initial_value))


def batch_norm(x):
    with tf.variable_scope('BN'):
        inputs_shape = x.get_shape()
        axis = list(range(len(inputs_shape) - 1))
        param_shape = inputs_shape[-1:]

        beta = tf.get_variable('beta', param_shape, initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable('gamma', param_shape, initializer=tf.constant_initializer(1.))
        batch_mean, batch_var = tf.nn.moments(x, axis)
        normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
    return normed


def dropout(x, keep_prob, is_training):
    return tf.cond(is_training, lambda: tf.nn.dropout(x, keep_prob), lambda: x)


def conv(x, filter, is_training):
    l = tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME')
    l = batch_norm(l, is_training)
    return tf.nn.relu(l)


def flatten(x):
    return tf.reshape(x, [-1])


def fully_connected(input, num_neurons, name, activation='relu', bn=True):
    func = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh}
    W = weight(name + '_W', [input.get_shape().as_list()[1], num_neurons], init='he')
    l = tf.matmul(input, W) + bias(name + '_b', num_neurons)
    if bn:
        l = batch_norm(l)
    return func[activation](l)
