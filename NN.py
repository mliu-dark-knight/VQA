import math
import tensorflow as tf


def weight(name, shape, init='he'):
	assert init == 'he'
	std = math.sqrt(2.0 / reduce(lambda x, y: x + y, shape[:-1]))
	initializer = tf.random_normal_initializer(stddev=std)

	var = tf.get_variable(name, shape, initializer=initializer)
	tf.add_to_collection('l2', tf.nn.l2_loss(var))
	return var


def bias(name, dim, initial_value=1e-2):
	return tf.get_variable(name, dim, initializer=tf.constant_initializer(initial_value))


def batch_norm(x, prefix):
	with tf.variable_scope('BN'):
		inputs_shape = x.get_shape()
		axis = list(range(len(inputs_shape) - 1))
		param_shape = inputs_shape[-1:]

		beta = tf.get_variable(prefix + '_beta', param_shape, initializer=tf.constant_initializer(0.))
		gamma = tf.get_variable(prefix + '_gamma', param_shape, initializer=tf.constant_initializer(1.))
		batch_mean, batch_var = tf.nn.moments(x, axis)
		normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
	return normed


def dropout(x, keep_prob, is_training):
	return tf.cond(is_training, lambda: tf.nn.dropout(x, keep_prob), lambda: x * keep_prob)



def conv1d(x, shape, stride, prefix, suffix='', activation='relu', bn=False):
	func = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'sigmoid': tf.nn.sigmoid, None: tf.identity}
	W = weight(prefix + '_W' + str(suffix), shape)
	if bn:
		l = batch_norm(tf.nn.conv1d(x, W, stride, padding='SAME'), prefix)
	else:
		b = bias(prefix + '_b' + str(suffix), shape[-1])
		l = tf.nn.conv1d(x, W, stride, padding='SAME') + b
	return func[activation](l)


def fully_connected(input, num_neurons, prefix, suffix='', activation='relu', bn=False):
	func = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'sigmoid': tf.nn.sigmoid, None: tf.identity}
	W = weight(prefix + '_W' + suffix, [input.get_shape().as_list()[1], num_neurons], init='he')
	if bn:
		l = batch_norm(tf.matmul(input, W), prefix)
	else:
		l = tf.matmul(input, W) + bias(prefix + '_b' + suffix, num_neurons)
	return func[activation](l)
