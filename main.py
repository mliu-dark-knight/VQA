import datetime
import shlex
import subprocess, os

import tensorflow as tf
from DMN_plus import DMN
from utils import *

flags = tf.app.flags

# directories
flags.DEFINE_boolean('test', False, 'true for testing, false for training')
#flags.DEFINE_string('save_dir', 'model/', 'Save path [save/]')

# training options
flags.DEFINE_integer('batch_size', 32, 'Batch size during training and testing')
flags.DEFINE_integer('dataset_size', 100, 'Maximum data point')
flags.DEFINE_integer('num_epochs', 100000, 'Number of epochs for training')
flags.DEFINE_integer('num_steps', 10, 'Number of steps per epoch')
flags.DEFINE_boolean('load', False, 'Start training from saved model')
flags.DEFINE_integer('save_period', 3, 'Save period [80]')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_float('decay_rate', 0.01, 'Decay rate')

# model params
flags.DEFINE_integer('memory_step', 3, 'Episodic Memory steps')
flags.DEFINE_string('memory_update', 'relu', 'Episodic meory update method - relu or gru')
flags.DEFINE_integer('glove_dim', 50, 'GloVe size - Only used in dmn')
flags.DEFINE_integer('vocab_size', 400000, 'Vocabulary size')
flags.DEFINE_integer('hidden_dim', 100, 'Size of hidden units')
flags.DEFINE_integer('channel_dim', 512, 'Number of channels')
flags.DEFINE_integer('img_size', 7 * 7, 'Image feature size')
flags.DEFINE_string('attention', 'soft', 'Attention mechanism')
flags.DEFINE_float('epsilon', 0.01, 'Annealing parameter for attention softmax')
flags.DEFINE_integer('max_ques_size', 25, 'Max length of question')
flags.DEFINE_float('lambda_r', 0.0, 'Regularization weight')
flags.DEFINE_float('lambda_t', 0.0, 'Question type weight')
flags.DEFINE_boolean('quasi_rnn', True, 'Use quasi rnn')
flags.DEFINE_integer('kernel_width', 2, 'Kernel width')
flags.DEFINE_string('pooling', '_ifo', 'Pooling method for quasi rnn, _f, _fo or _ifo')
flags.DEFINE_integer('rnn_layer', 2, 'Number of layers in question encoder')


FLAGS = flags.FLAGS

def main(_):
	try:
		# FLAGS.attention = FLAGS_cmd.attn
		# FLAGS.hidden_dim = FLAGS_cmd.hidden_dim
		# FLAGS.memory_update = FLAGS_cmd.memory_update
		# FLAGS.pooling = FLAGS_cmd.pooling
		# FLAGS.rnn_layer = FLAGS_cmd.rnn_layer

		# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

		FLAGS.save_dir = "model_" + FLAGS.attention + "_" + str(FLAGS.hidden_dim) + "_" + \
						 str(FLAGS.memory_update) + "_" + FLAGS.pooling + "_" + \
						 datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + "/"

		tboard_proc = subprocess.Popen(shlex.split(
			'/home/victor/anaconda3/bin/tensorboard --logdir=' + FLAGS.save_dir + 'Log'))

		os.makedirs(FLAGS.save_dir)
		word2vec = WordTable(FLAGS.glove_dim)

		FLAGS.vocab_size = word2vec.vocab_size
		dataset = DataSet(word2vec=word2vec, params=FLAGS, type='train', q_max=30, num_threads=8)
		#val_dataset = DataSet(word2vec=word2vec, params=FLAGS, type='val', q_max=3, num_threads=1)
		with tf.Session() as sess:
			model = DMN(FLAGS, None)
			summary_writer = tf.summary.FileWriter(FLAGS.save_dir + 'log', graph=sess.graph)
			sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
			model.train(sess, dataset, dataset, summary_writer)
	finally:
		try:
			dataset.kill()
			#val_dataset.kill()
			tboard_proc.terminate()
		except NameError:
			pass

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument(
	# 	'attn',
	# 	type=str,
	# 	help='soft or gru'
	# )
	# parser.add_argument(
	# 	'memory_update',
	# 	type=str,
	# 	help='relu or gru'
	# )
	# parser.add_argument(
	# 	'quasi_rnn',
	# 	type=bool,
	# 	help='T or F'
	# )
	# parser.add_argument(
	# 	'pooling',
	# 	type=str,
	# 	help='_fo or _ifo'
	# )
	# parser.add_argument(
	# 	'hidden_dim',
	# 	type=int,
	# 	help='50-5000'
	# )
	# # parser.add_argument(
	# # 	'rnn_layer',
	# # 	type=int,
	# # 	help='batch_size'
	# # )
	# FLAGS_cmd, unparsed = parser.parse_known_args()
	tf.app.run()
