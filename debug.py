import numpy as np
import tensorflow as tf
from DMN_plus import DMN

flags = tf.app.flags

# directories
flags.DEFINE_boolean('test', False, 'true for testing, false for training')
flags.DEFINE_string('save_dir', 'model/log/', 'Save path')

# training options
flags.DEFINE_integer('batch_size', 1, 'Batch size during training and testing')
flags.DEFINE_integer('dataset_size', 1, 'Maximum data point')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs for training')
flags.DEFINE_integer('num_steps', 1, 'Number of steps per epoch')
flags.DEFINE_boolean('load', False, 'Start training from saved model')
flags.DEFINE_integer('save_period', 3, 'Save period [80]')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_float('decay_rate', 0.1, 'Decay rate')

# model params
flags.DEFINE_integer('num_ques_type', 3, 'Number of types of question')
flags.DEFINE_integer('memory_step', 1, 'Episodic Memory steps')
flags.DEFINE_string('memory_update', 'relu', 'Episodic meory update method - relu or gru')
flags.DEFINE_integer('glove_dim', 5, 'GloVe size - Only used in dmn')
flags.DEFINE_integer('num_range', 10, 'Range of numerical answer')
flags.DEFINE_integer('vocab_size', 40, 'Vocabulary size')
flags.DEFINE_integer('hidden_dim', 10, 'Size of hidden units')
flags.DEFINE_integer('channel_dim', 16, 'Number of channels')
flags.DEFINE_integer('img_size', 3 * 3, 'Image feature size')
flags.DEFINE_string('attention', 'soft', 'Attention mechanism, soft or gru')
flags.DEFINE_float('epsilon', 0.01, 'Annealing parameter for attention softmax')
flags.DEFINE_integer('max_ques_size', 10, 'Max length of question')
flags.DEFINE_float('lambda_r', 0.0, 'Regularization weight')
flags.DEFINE_float('lambda_t', 0.0, 'Question type weight')
flags.DEFINE_boolean('quasi_rnn', True, 'Use quasi rnn')
flags.DEFINE_integer('kernel_width', 2, 'Kernel width')
flags.DEFINE_string('pooling', '_ifo', 'Pooling method for quasi rnn, _f, _fo or _ifo')
flags.DEFINE_integer('rnn_layer', 1, 'Number of layers in question encoder')


FLAGS = flags.FLAGS

class FakeDataSet(object):
    def __init__(self):
        pass

    def index_to_word(self, index):
        return 'invalid'

    def next_batch(self):
        return [np.array([None]), np.array([None]), np.random.rand(FLAGS.batch_size, FLAGS.img_size, FLAGS.channel_dim),
                np.random.rand(FLAGS.batch_size, FLAGS.max_ques_size, FLAGS.glove_dim), np.random.randint(2, size=FLAGS.batch_size),
                np.array([None]), np.array([None]), np.random.rand(FLAGS.batch_size, FLAGS.img_size, FLAGS.channel_dim),
                np.random.rand(FLAGS.batch_size, FLAGS.max_ques_size, FLAGS.glove_dim), np.random.randint(FLAGS.num_range, size=FLAGS.batch_size),
                np.array([None]), np.array([None]), np.random.rand(FLAGS.batch_size, FLAGS.img_size, FLAGS.channel_dim),
                np.random.rand(FLAGS.batch_size, FLAGS.max_ques_size, FLAGS.glove_dim), np.random.randint(FLAGS.vocab_size, size=FLAGS.batch_size)]



def main(_):
    dataset = FakeDataSet()
    with tf.Session() as sess:
        model = DMN(FLAGS, None)
        summary_writer = tf.summary.FileWriter(FLAGS.save_dir, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        model.train(sess, dataset, dataset, summary_writer)

if __name__ == '__main__':
	tf.app.run()
