from __future__ import print_function
import datetime
import sys
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class Base(object):
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
            self.merged_summary_op = tf.summary.merge_all()
            self.merged_summary_op_b_loss = tf.summary.merge_all(key='b_stuff')
            self.merged_summary_op_n_loss = tf.summary.merge_all(key='n_stuff')
            self.merged_summary_op_m_loss = tf.summary.merge_all(key='m_stuff')


    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, type, sess):
        if type == 'b':
            (_, Is, Xs, Qs, As, _, _, _, _, _, _, _, _, _, _) = batch
            type = np.zeros_like(As)
            answer = self.answer_b
        elif type == 'n':
            (_, _, _, _, _, _, Is, Xs, Qs, As, _, _, _, _, _) = batch
            type = np.ones_like(As)
            answer = self.answer_n
        else:
            (_, _, _, _, _, _, _, _, _, _, _, Is, Xs, Qs, As) = batch
            type = np.repeat(2, len(As))
            answer = self.answer_m
        return {self.input: Xs, self.question: Qs, self.type: type, answer: As}

    def train_batch(self, sess, batch, sum_writer):
        for (type, gradient_descent, summary_op_for_that_type) in [
            ('b', self.gradient_descent_b, self.merged_summary_op_b_loss),
            ('n', self.gradient_descent_n, self.merged_summary_op_n_loss),
            ('m', self.gradient_descent_m, self.merged_summary_op_m_loss)]:
            feed_dict = self.get_feed_dict(batch, type, sess)
      #      gradient = sess.run(getattr(self, 'debug_' + type), feed_dict=feed_dict)
#            print(gradient)
            if len(feed_dict[self.type]) > 0:
                _, global_step, summary_all, specialized_summary = sess.run(
                    [gradient_descent, self.global_step, self.merged_summary_op, summary_op_for_that_type],
                    feed_dict=feed_dict) #  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            #run_metadata=run_metadata)
                sum_writer.add_summary(summary_all, global_step=global_step)
                sum_writer.add_summary(specialized_summary, global_step=global_step)

    def test_batch(self, sess, batch):
        ret_list = []
        for (type, predicts, accuracy) in \
                [('b', self.predicts_b, self.accuracy_b), ('n', self.predicts_n, self.accuracy_n),
                 ('m', self.predicts_m, self.accuracy_m)]:
            feed_dict = self.get_feed_dict(batch, type, sess)
            if len(feed_dict[self.type]) > 0:
                ret_list += sess.run([self.predicts_t, predicts, accuracy], feed_dict=feed_dict)
            else:
                # accuracy -1.0 means empty test data
                ret_list += [[], [], -1.0]
        return ret_list

    def train(self, sess, train_data, val_data, sum_writer):
        for epoch in tqdm(range(self.params.num_epochs), desc='Epoch', maxinterval=10000, ncols=100):
            for step in tqdm(range(self.params.num_steps), desc='Step', maxinterval=10000, ncols=100):
                batch = train_data.next_batch()
                self.train_batch(sess, batch, sum_writer)

            self.eval(sess, val_data)
            if epoch % self.params.save_period == 0:
                self.save(sess)
                print('Saved! Current Epoch: ' + str(epoch) + ', Current Step: ' + str(step) + ", Total Time: " +
                      str(datetime.timedelta(seconds=(time.time() - self.starttime_init))))

        print('Training complete.')

    def eval(self, sess, eval_data):
        batch = eval_data.next_batch()
        predicts_tb, predicts_b, accuracy_b, predicts_tn, predicts_n, accuracy_n, predicts_tm, predicts_m, accuracy_m = \
            self.test_batch(sess, batch)
        (Anns_b, Is_b, _, _, _, Anns_n, Is_n, _, _, _, Anns_m, Is_m, _, _, _) = batch
        for predict, Ann, I in zip(predicts_b, Anns_b, Is_b):
           # eval_data.visualize(Ann, I)
            tqdm.write('Predicted answer: %s' % ('yes' if predict == 1 else 'no'))
        tqdm.write('Accuracy (yes/no): %f' % accuracy_b)
        for predict, Ann, I in zip(predicts_b, Anns_n, Is_n):
           # eval_data.visualize(Ann, I)
            tqdm.write('Predicted answer: %d' % (predict))
        tqdm.write('Accuracy (how many): %f' % accuracy_n)
        for predict, Ann, I in zip(predicts_m, Anns_m, Is_m):
           # eval_data.visualize(Ann, I)
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
