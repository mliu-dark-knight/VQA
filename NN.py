import math
import tensorflow as tf
from functools import *


def weight(name, shape, init='he', type=None):
	assert init == 'he'
	std = math.sqrt(2.0 / reduce(lambda x, y: x + y, [0] + shape[:-1]))
	initializer = tf.random_normal_initializer(stddev=std)
	if type is not None:
		var = tf.get_variable(name, shape, initializer=initializer, trainable=False)
		tf.add_to_collection(type, var)
	else:
		var = tf.get_variable(name, shape, initializer=initializer)

	tf.add_to_collection('l2', tf.nn.l2_loss(var))
	return var


def embedding(name, shape):
	return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(minval=-1.0 / shape[1], maxval=1.0 / shape[1]))


def bias(name, dim, initial_value=1e-2, type=None):
	if type is not None:
		var = tf.get_variable(name, dim, initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_OUT'), trainable=False)
		tf.add_to_collection(type, var)
		return var
	else:
		return tf.get_variable(name, dim, initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_OUT'))


def batch_norm(x, prefix, training):
	with tf.variable_scope('BN'):
		inputs_shape = x.get_shape()
		axis = list(range(len(inputs_shape) - 1))
		param_shape = inputs_shape[-1:]

		beta = tf.get_variable(prefix + '_beta', param_shape, initializer=tf.constant_initializer(0.))
		gamma = tf.get_variable(prefix + '_gamma', param_shape, initializer=tf.constant_initializer(1.))
		batch_mean, batch_var = tf.nn.moments(x, axis)
		ema = tf.train.ExponentialMovingAverage(decay=0.9)

		def update_mean_var():
			ema.apply([batch_mean, batch_var])
			return batch_mean, batch_var

		mean, var = tf.cond(training, update_mean_var, lambda : (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed


def dropout(x, keep_prob, training):
	return tf.cond(training, lambda: tf.nn.dropout(x, keep_prob), lambda: x * keep_prob)



def conv1d(x, shape, stride, prefix, suffix='', activation='relu', bn=False):
	func = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'sigmoid': tf.nn.sigmoid, None: tf.identity}
	W = weight(prefix + '_W' + str(suffix), shape)
	if bn:
		l = batch_norm(tf.nn.conv1d(x, W, stride, padding='SAME'), prefix)
	else:
		b = bias(prefix + '_b' + str(suffix), shape[-1])
		l = tf.nn.conv1d(x, W, stride, padding='SAME') + b
	return func[activation](l)


def fully_connected(input, num_neurons, prefix, suffix='', activation='relu', bn=False, training=True, type=None):
	func = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'sigmoid': tf.nn.sigmoid, None: tf.identity}
	W = weight(prefix + '_W' + suffix, [input.get_shape().as_list()[1], num_neurons], init='he', type=type)
	if bn:
		l = batch_norm(tf.matmul(input, W), prefix, training)
	else:
		l = tf.matmul(input, W) + bias(prefix + '_b' + suffix, num_neurons)
	return func[activation](l)
