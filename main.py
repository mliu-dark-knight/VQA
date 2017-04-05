import os
import tensorflow as tf
from DMN_plus import DMN
from utils import *

flags = tf.app.flags

# directories
flags.DEFINE_string('model', 'DMN+', 'Model type - DMN+, DMN, [Default: DMN+]')
flags.DEFINE_boolean('test', False, 'true for testing, false for training [False]')
flags.DEFINE_string('save_dir', 'model/', 'Save path [save/]')

# training options
flags.DEFINE_integer('batch_size', 10, 'Batch size during training and testing [128]')
flags.DEFINE_integer('dataset_size', 10, 'Maximum data point')
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs for training [256]')
flags.DEFINE_integer('num_steps', 10, 'Number of steps per epoch')
flags.DEFINE_boolean('load', False, 'Start training from saved model? [False]')
flags.DEFINE_integer('save_period', 80, 'Save period [80]')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_float('decay_rate', 0.1, 'Decay rate')

# model params
flags.DEFINE_integer('memory_step', 3, 'Episodic Memory steps')
flags.DEFINE_string('memory_update', 'relu', 'Episodic meory update method - relu or gru')
flags.DEFINE_integer('glove_dim', 50, 'GloVe size - Only used in dmn')
flags.DEFINE_integer('vocab_size', 400000, 'Vocabulary size')
flags.DEFINE_integer('hidden_dim', 80, 'Size of hidden units')
flags.DEFINE_integer('channel_dim', 512, 'Number of channels')
flags.DEFINE_integer('img_size', 7 * 7, 'Image feature size')
flags.DEFINE_string('attention', 'gru', 'Attention mechanism')
flags.DEFINE_integer('max_ques_size', 10, 'Max length of question')
flags.DEFINE_float('lambda_r', 0.0, 'Regularization weight')
flags.DEFINE_float('lambda_t', 0.0, 'Question type weight')
flags.DEFINE_boolean('quasi_rnn', True, 'Use quasi rnn')
flags.DEFINE_integer('kernel_width', 2, 'Kernel width')
flags.DEFINE_string('pooling', '_ifo', 'Pooling method for quasi rnn, _f, _fo or _ifo')
flags.DEFINE_integer('rnn_layer', 2, 'Number of layers in question encoder')


FLAGS = flags.FLAGS


def main(_):
	word2vec = WordTable(FLAGS.glove_dim)
	FLAGS.vocab_size = word2vec.vocab_size
	dataset = DataSet(word2vec=word2vec, params=FLAGS)
	with tf.Session() as sess:
		model = DMN(FLAGS, None)
		sess.run(tf.global_variables_initializer())
		summary_writer = tf.summary.FileWriter('log', graph=sess.graph)
		model.train(sess, dataset)

if __name__ == '__main__':
	tf.app.run()
