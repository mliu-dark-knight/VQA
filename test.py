import os
import tensorflow as tf
from DMN_plus import DMN
from utils import *

flags = tf.app.flags

# directories
flags.DEFINE_string('model', 'DMN+', 'Model type - DMN+, DMN, [Default: DMN+]')
flags.DEFINE_boolean('test', False, 'true for testing, false for training [False]')
flags.DEFINE_string('data_dir', 'data/tasks_1-20_v1-2/en-10k', 'Data directory [data/tasks_1-20_v1-2/en-10k]')
flags.DEFINE_string('save_dir', 'save', 'Save path [save]')

# training options
flags.DEFINE_bool('gpu', True, 'Use GPU? [True]')
flags.DEFINE_integer('batch_size', 128, 'Batch size during training and testing [128]')
flags.DEFINE_integer('num_epochs', 256, 'Number of epochs for training [256]')
flags.DEFINE_float('learning_rate', 0.002, 'Learning rate [0.002]')
flags.DEFINE_boolean('load', False, 'Start training from saved model? [False]')
flags.DEFINE_integer('acc_period', 10, 'Training accuracy display period [10]')
flags.DEFINE_integer('val_period', 40, 'Validation period (for display purpose) [40]')
flags.DEFINE_integer('save_period', 80, 'Save period [80]')

# model params
flags.DEFINE_integer('memory_step', 3, 'Episodic Memory steps [3]')
flags.DEFINE_string('memory_update', 'relu', 'Episodic meory update method - relu or gru [relu]')
# flags.DEFINE_bool('memory_tied', False, 'Share memory update weights among the layers? [False]')
flags.DEFINE_integer('glove_dim', 50, 'GloVe size - Only used in dmn [50]')
flags.DEFINE_integer('vocab_size', 400000, 'Vocabulary size, delete this line during training')
flags.DEFINE_integer('embed_size', 80, 'Word embedding size - Used in dmn+, dmn_embed [80]')
flags.DEFINE_integer('hidden_dim', 80, 'Size of hidden units [80]')
flags.DEFINE_integer('channel_dim', 512, 'Number of channels')
flags.DEFINE_integer('img_size', 15 * 20, 'Image feature size')
flags.DEFINE_integer('max_ques_size', 20, 'Max length of question')

# train hyperparameters
flags.DEFINE_float('weight_decay', 0.001, 'Weight decay - 0 to turn off L2 regularization [0.001]')
flags.DEFINE_float('keep_prob', 1., 'Dropout rate - 1.0 to turn off [1.0]')
flags.DEFINE_bool('batch_norm', True, 'Use batch normalization? [True]')

# bAbi dataset params
flags.DEFINE_integer('task', 1, 'bAbi Task number [1]')
flags.DEFINE_float('val_ratio', 0.1, 'Validation data ratio to training data [0.1]')

FLAGS = flags.FLAGS


def main(_):
    with tf.Session() as sess:
        model = DMN(FLAGS, None)

if __name__ == '__main__':
    tf.app.run()
