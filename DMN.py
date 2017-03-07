import sys
import tqdm
import numpy as np
import tensorflow as tf


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
        return sess.run([self.accuracy, self.total_loss], feed_dict=feed_dict)

    def train(self, sess, train_data):
        for epoch in range(self.params.num_epochs):
            for step in range(self.params.num_steps):
                batch = train_data.next_batch()
                self.train_batch(sess, batch)

            batch = train_data.next_batch()
            accuracy, _ = self.test_batch(sess, batch)
            print('Accuracy: %f' % accuracy)
            if epoch % self.params.save_period == 0:
                self.save(sess)

        print('Training complete.')

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
