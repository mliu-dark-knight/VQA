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
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training [256]')
flags.DEFINE_integer('num_steps', 10, 'Number of steps per epoch')
flags.DEFINE_boolean('load', False, 'Start training from saved model? [False]')
flags.DEFINE_integer('save_period', 80, 'Save period [80]')

# model params
flags.DEFINE_integer('memory_step', 3, 'Episodic Memory steps [3]')
flags.DEFINE_string('memory_update', 'relu', 'Episodic meory update method - relu or gru [relu]')
# flags.DEFINE_bool('memory_tied', False, 'Share memory update weights among the layers? [False]')
flags.DEFINE_integer('glove_dim', 50, 'GloVe size - Only used in dmn [50]')
flags.DEFINE_integer('vocab_size', 400000, 'Vocabulary size, delete this line during training')
flags.DEFINE_integer('embed_size', 100, 'Word embedding size - Used in dmn+, dmn_embed [80]')
flags.DEFINE_integer('hidden_dim', 80, 'Size of hidden units [80]')
flags.DEFINE_integer('channel_dim', 512, 'Number of channels')
flags.DEFINE_integer('img_size', 7 * 7, 'Image feature size')
flags.DEFINE_bool('episode_memory', False, 'Use episode memory')
flags.DEFINE_bool('question_coattention', True, 'Use question coattention')
flags.DEFINE_integer('max_ques_size', 10, 'Max length of question, [None] for dynamic rnn')

# train hyperparameters
flags.DEFINE_float('weight_decay', 0.001, 'Weight decay - 0 to turn off L2 regularization [0.001]')
flags.DEFINE_float('keep_prob', 1., 'Dropout rate - 1.0 to turn off [1.0]')
flags.DEFINE_bool('batch_norm', True, 'Use batch normalization? [True]')

# bAbi dataset params
flags.DEFINE_integer('task', 1, 'bAbi Task number [1]')
flags.DEFINE_float('val_ratio', 0.1, 'Validation data ratio to training data [0.1]')

FLAGS = flags.FLAGS


def main(_):
	word2vec = WordTable(FLAGS.glove_dim)
	FLAGS.vocab_size = word2vec.vocab_size
	dataset = DataSet(word2vec=word2vec, max_ques_size=FLAGS.max_ques_size, batch_size=FLAGS.batch_size, dataset_size=FLAGS.dataset_size)
	with tf.Session() as sess:
		model = DMN(FLAGS, None)
		sess.run(tf.global_variables_initializer())
		summary_writer = tf.summary.FileWriter('log', graph=sess.graph)
		model.train(sess, dataset)

if __name__ == '__main__':
	tf.app.run()
