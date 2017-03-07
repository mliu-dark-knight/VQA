import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class BaseModel(object):
    def __init__(self, params, words):
        self.params = params
        self.words = words
        self.save_dir = params.save_dir

        with tf.variable_scope('DMN'):
            print("Building DMN...")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            self.build()
            self.saver = tf.train.Saver()
            self.merged_summary_op = tf.summary.merge_all()


    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, sess):
        def learning_rate(step):
            return 1.0 / (1.0 + step * 1e-2)

        (Is, Xs, Qs, As) = batch
        return {self.input: Xs, self.question: Qs, self.answer: As, self.learning_rate: learning_rate(sess.run(self.global_step))}

    def train_batch(self, sess, batch):
        feed_dict = self.get_feed_dict(batch, sess)
        sess.run([self.gradient_descent, self.global_step], feed_dict=feed_dict)

    def test_batch(self, sess, batch):
        feed_dict = self.get_feed_dict(batch, sess)
        return sess.run([self.predicts, self.accuracy], feed_dict=feed_dict)

    def train(self, sess, train_data):
        for epoch in tqdm(range(self.params.num_epochs), desc='Epoch', maxinterval=10000, ncols=100):
            for step in tqdm(range(self.params.num_steps), desc='Step', maxinterval=10000, ncols=100):
                batch = train_data.next_batch()
                self.train_batch(sess, batch[1:])

            self.eval(sess, train_data)
            if epoch % self.params.save_period == 0:
                self.save(sess)

        print('Training complete.')

    def eval(self, sess, eval_data):
        batch = eval_data.next_batch()
        predicts, accuracy = self.test_batch(sess, batch[1:])
        (Anns, Is, _, _, _) = batch
        for predict, Ann, I in zip(predicts, Anns, Is):
            eval_data.visualize(Ann, I)
            tqdm.write('Predicted answer: %s' % eval_data.index_to_word(predict))
        tqdm.write('Accuracy: %f' % accuracy)

    def save(self, sess):
        print('Saving model to %s' % self.save_dir)
        self.saver.save(sess, self.save_dir, self.global_step)

    def load(self, sess):
        print('Loading model ...')
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint is None:
            print('Error: No saved model found')
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)
