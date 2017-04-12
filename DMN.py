from __future__ import print_function
import sys, datetime, time
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class BaseModel(object):
    def __init__(self, params, words):
        self.params = params
        self.words = words
        self.save_dir = params.save_dir
        self.starttime_init = time.time()

        with tf.variable_scope('DMN'):
            print("Building DMN...")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.build()
            self.saver = tf.train.Saver()

            # for var in tf.trainable_variables():
            #     tf.summary.histogram(var.name, var)
            # self.merged_summary_op = tf.summary.merge_all()


    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, type, sess):
        if type == 'b':
            (_, Is, Xs, Qs, As, _, _, _, _, _) = batch
            type = np.zeros_like(As)
            answer = self.answer_b
        else:
            (_, _, _, _, _, _, Is, Xs, Qs, As) = batch
            type = np.ones_like(As)
            answer = self.answer_m
        return {self.input: Xs, self.question: Qs, self.type: type, answer: As}

    def train_batch(self, sess, batch):
        for (type, gradient_descent) in [('b', self.gradient_descent_b), ('m', self.gradient_descent_m)]:
            feed_dict = self.get_feed_dict(batch, type, sess)
            gradients = sess.run(self.debug, feed_dict=feed_dict)
            print('gradients:')
            print(gradients)
            if len(feed_dict[self.type]) > 0:
                sess.run([gradient_descent, self.global_step], feed_dict=feed_dict)

    def test_batch(self, sess, batch):
        ret_list = []
        for (type, predicts, accuracy) in [('b', self.predicts_b, self.accuracy_b), ('m', self.predicts_m, self.accuracy_m)]:
            feed_dict = self.get_feed_dict(batch, type, sess)
            if len(feed_dict[self.type]) > 0:
                ret_list += sess.run([self.predicts_t, predicts, accuracy], feed_dict=feed_dict)
            else:
                ret_list += [[], [], 0.0]
        return ret_list

    def train(self, sess, train_data, val_data):
        for epoch in tqdm(range(self.params.num_epochs), desc='Epoch', maxinterval=10000, ncols=100):
            for step in tqdm(range(self.params.num_steps), desc='Step', maxinterval=10000, ncols=100):
                batch = train_data.next_batch()
                self.train_batch(sess, batch)

            self.eval(sess, val_data)
            if epoch % self.params.save_period == 0:
                self.save(sess)
                print('Saved! Current Epoch: ' + str(epoch) + ', Current Step: ' + str(step) + ", Total Time: " +
                      str(datetime.timedelta(seconds=(time.time() - self.starttime_init))))

        print('Training complete.')

    def eval(self, sess, eval_data):
        batch = eval_data.next_batch()
        predicts_tb, predicts_b, accuracy_b, predicts_tm, predicts_m, accuracy_m = self.test_batch(sess, batch)
        (Anns_b, Is_b, _, _, _, Anns_m, Is_m, _, _, _) = batch
        for predict, Ann, I in zip(predicts_b, Anns_b, Is_b):
            #eval_data.visualize(Ann, I)
            tqdm.write('Predicted answer: %s' % ('yes' if predict == 1 else 'no'))
        tqdm.write('Accuracy (yes/no): %f' % accuracy_b)
        for predict, Ann, I in zip(predicts_m, Anns_m, Is_m):
            #eval_data.visualize(Ann, I)
            tqdm.write('Predicted answer: %s' % eval_data.index_to_word(predict))
        tqdm.write('Accuracy (other): %f' % accuracy_m)


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
