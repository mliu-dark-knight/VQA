import datetime
import shlex
import subprocess, os

import tensorflow as tf
from DMN_plus import DMN
from utils import *

flags = tf.app.flags

# directories
flags.DEFINE_boolean('test', False, 'true for testing, false for training')
flags.DEFINE_string('save_dir', 'model/log/', 'Save path [save/]')

# training options
flags.DEFINE_integer('batch_size', 1, 'Batch size during training and testing')
flags.DEFINE_integer('dataset_size', 100, 'Maximum data point')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs for training')
flags.DEFINE_integer('num_steps', 1, 'Number of steps per epoch')
flags.DEFINE_boolean('load', False, 'Start training from saved model')
flags.DEFINE_integer('save_period', 10, 'Save period [80]')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_float('decay_rate', 0.01, 'Decay rate')

# model params
flags.DEFINE_integer('num_ques_type', 4, 'Number of types of question')
flags.DEFINE_integer('memory_step', 3, 'Episodic Memory steps')
flags.DEFINE_string('memory_update', 'relu', 'Episodic meory update method - relu or gru')
flags.DEFINE_integer('glove_dim', 50, 'GloVe size - Only used in dmn')
flags.DEFINE_integer('vocab_size', 400000, 'Vocabulary size')
flags.DEFINE_integer('num_color', 20, 'Number of colors')
flags.DEFINE_integer('num_range', 20, 'Range of numerical answer')
flags.DEFINE_integer('hidden_dim', 100, 'Size of hidden units')
flags.DEFINE_integer('channel_dim', 512, 'Number of channels')
flags.DEFINE_integer('img_size', 7 * 7, 'Image feature size')
flags.DEFINE_string('attention', 'soft', 'Attention mechanism')
flags.DEFINE_float('epsilon', 0.01, 'Annealing parameter for attention softmax')
flags.DEFINE_integer('max_ques_size', 10, 'Max length of question')
flags.DEFINE_float('lambda_r', 0.1, 'Regularization weight')
flags.DEFINE_float('lambda_t', 0.1, 'Question type weight')
flags.DEFINE_boolean('quasi_rnn', True, 'Use quasi rnn')
flags.DEFINE_integer('kernel_width', 2, 'Kernel width')
flags.DEFINE_string('pooling', '_ifo', 'Pooling method for quasi rnn, _f, _fo or _ifo')
flags.DEFINE_integer('rnn_layer', 2, 'Number of layers in question encoder')


FLAGS = flags.FLAGS

def main(_):
	word2vec = WordTable.load_word2vec()
	FLAGS.vocab_size = word2vec.vocab_size
	dataset = DataSet(word2vec=word2vec, params=FLAGS, type='train', q_max=1, num_threads=1)
	with tf.Session() as sess:
		model = DMN(FLAGS)
		summary_writer = tf.summary.FileWriter(FLAGS.save_dir, graph=sess.graph)
		sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
		model.train(sess, dataset, dataset, summary_writer)
	dataset.kill()

if __name__ == '__main__':
	tf.app.run()
