from __future__ import print_function
from DMN import BaseModel
from NN import *


class AttentionGRU(object):
	def __init__(self, num_units, input_dim):
		self.num_units = num_units
		self.input_dim = input_dim

	def __call__(self, inputs, state, attention):
		with tf.name_scope('Attention_GRU'):
			r = tf.nn.sigmoid(self.linear(inputs, state, name='r', bias_default=1.0))
			h = tf.nn.tanh(self.linear(inputs, r * state, name='h'))
			new_h = attention * h + (1 - attention) * state
		return new_h

	def linear(self, x, h, name, bias_default=0.0):
		W = weight('AttentionGRU_W' + name, [self.input_dim, self.num_units])
		U = weight('AttentionGRU_U' + name, [self.num_units, self.num_units])
		b = bias('AttentionGRU_b' + name, self.num_units, bias_default)
		return tf.matmul(x, W) + tf.matmul(h, U) + b


class EpisodeMemory:
	def __init__(self, hidden_dim, facts, attention):
		self.hidden_dim = hidden_dim
		self.facts = tf.unstack(facts, axis=1)

		self.W1 = weight('EpisodeMemory_W1', [4 * hidden_dim, hidden_dim])
		self.b1 = bias('EpisodeMemory_b1', [hidden_dim])
		self.W2 = weight('EpisodeMemory_W2', [hidden_dim, 1])
		self.b2 = bias('EpisodeMemory_b2', [1])
		if attention == 'gru':
			self.gru = AttentionGRU(self.hidden_dim, self.hidden_dim)

	def init_hidden(self, question):
		return tf.zeros_like(question)

	def update(self, memory, question, attention):
		gs = self.attention(self.facts, memory, question)
		if attention == 'soft':
			facts = tf.stack(self.facts, axis=1)
			return tf.transpose(tf.reduce_sum(tf.transpose(facts, perm=[2, 1, 0]) * gs, axis=1))

		else:
			gs = tf.unstack(gs, axis=1)
			hidden = self.init_hidden(question)
			with tf.variable_scope('Attention_Gate') as scope:
				for f, g in zip(self.facts, gs):
					hidden = self.gru(f, hidden, g)
					scope.reuse_variables()
			return hidden

	def attention(self, fs, m, q):
		with tf.name_scope('Attention'):
			Z = []
			for f in fs:
				z = tf.concat([f * q, f * m, tf.abs(f - q), tf.abs(f - m)], 1)
				Z.append(tf.matmul(tf.nn.tanh(tf.matmul(z, self.W1) + self.b1), self.W2) + self.b2)
			g = tf.nn.softmax(tf.stack(Z, axis=1))
		return g

'''
Do not feed this into MultiRNN, for performance sake
'''
class QuasiRNN(object):
	counter = 0
	def __init__(self, num_units, kernel_w):
		self.num_units = num_units
		self.kernel_w = kernel_w
		self.id = QuasiRNN.counter
		QuasiRNN.counter += 1

	def __call__(self, inputs, pooling):
		def _f(z, i, f, o, h, c):
			h = f * h + (1 - f) * z
			return f, c

		def _fo(z, i, f, o, h, c):
			c = f * c + (1 - f) * z
			h = o * c
			return h, c

		def _ifo(z, i, f, o, h, c):
			c = f * c + i * z
			h = o * c
			return h, c

		shape = [self.kernel_w] + inputs.get_shape().as_list()[2:] + [self.num_units]
		Z = tf.unstack(conv1d(inputs, shape, 1, 'Z', suffix=self.id, activation='tanh'), axis=1)
		I = tf.unstack(conv1d(inputs, shape, 1, 'I', suffix=self.id, activation='sigmoid'), axis=1)
		F = tf.unstack(conv1d(inputs, shape, 1, 'F', suffix=self.id, activation='sigmoid'), axis=1)
		O = tf.unstack(conv1d(inputs, shape, 1, 'O', suffix=self.id, activation='sigmoid'), axis=1)
		h, c = tf.zeros_like(O[0]), tf.zeros_like(O[0])
		H, C = [], []
		for z, i, f, o in zip(Z, I, F, O):
			h, c = eval(pooling)(z, i, f, o, h, c)
			H.append(h)
			C.append(c)
		return tf.stack(H, axis=1), tf.stack(H, axis=1)


class DMN(BaseModel):
	def build(self):
		self.input = tf.placeholder(tf.float32, shape=[None, self.params.img_size, self.params.channel_dim])
		self.question = tf.placeholder(tf.float32, shape=[None, self.params.max_ques_size, self.params.glove_dim])
		self.type = tf.placeholder(tf.int32, shape=[None])
		self.answer_b = tf.placeholder(tf.int32, shape=[None])
		self.answer_m = tf.placeholder(tf.int32, shape=[None])

		facts = self.build_input()
		questions = self.build_question()
		type = self.build_type(tf.unstack(questions, axis=1)[-1])
		memory = self.build_memory(questions, facts)
		logits_b = self.build_logits(memory, 2, 'AnswerBinary')
		logits_m = self.build_logits(memory, self.params.vocab_size, 'AnswerMulti')

		with tf.name_scope('Loss'):
			loss_t = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.type, logits=type))
			loss_b = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_b, logits=logits_b))
			loss_m = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_m, logits=logits_m))
			total_loss_b = loss_b + self.params.lambda_t * loss_t + self.params.lambda_r * tf.add_n(tf.get_collection('l2'))
			total_loss_m = loss_m + self.params.lambda_t * loss_t + self.params.lambda_r * tf.add_n(tf.get_collection('l2'))
			# tf.summary.scalar('Cross_Entropy', loss)

		with tf.name_scope('Accuracy'):
			self.predicts_t = tf.cast(tf.argmax(type, 1), 'int32')
			self.predicts_b = tf.cast(tf.argmax(logits_b, 1), 'int32')
			self.predicts_m = tf.cast(tf.argmax(logits_m, 1), 'int32')
			self.accuracy_t = tf.reduce_mean(tf.cast(tf.equal(self.predicts_t, self.type), tf.float32))
			self.accuracy_b = tf.reduce_mean(tf.cast(tf.equal(self.predicts_b, self.answer_b), tf.float32))
			self.accuracy_m = tf.reduce_mean(tf.cast(tf.equal(self.predicts_m, self.answer_m), tf.float32))

		learning_rate = tf.train.inverse_time_decay(self.params.learning_rate, self.global_step, 1, self.params.decay_rate)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.gradient_descent_b = optimizer.minimize(total_loss_b, global_step=self.global_step)
		self.gradient_descent_m = optimizer.minimize(total_loss_m, global_step=self.global_step)

		for variable in tf.trainable_variables():
			print(variable.name, variable.get_shape())

	def build_input(self):
		input = tf.reshape(self.input, [-1, self.params.channel_dim])
		facts = fully_connected(input, self.params.hidden_dim, 'ImageEmbedding', activation=None)
		return tf.reshape(facts, [-1, self.params.img_size, self.params.hidden_dim])

	def build_question(self):
		with tf.name_scope('Question_Embedding'):
			if self.params.quasi_rnn:
				rnn_inputs = self.question
				for _ in range(self.params.rnn_layer):
					rnn = QuasiRNN(self.params.hidden_dim, self.params.kernel_width)
					rnn_inputs, _ = rnn(rnn_inputs, self.params.pooling)
				question_vecs = rnn_inputs
			else:
				gru = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.params.hidden_dim)] * self.params.rnn_layer)
				question_vecs, _ = tf.nn.dynamic_rnn(gru, self.question, dtype=tf.float32)
		return question_vecs

	def build_type(self, question):
		with tf.name_scope('Question_Type'):
			return fully_connected(question, 2, 'Type', activation=None, bn=False)

	def build_memory(self, questions, facts):
		with tf.variable_scope('Memory') as scope:
			question = tf.identity(tf.unstack(questions, axis=1)[-1])
			episode = EpisodeMemory(self.params.hidden_dim, facts, self.params.attention)
			memory = tf.identity(question)
			gru = tf.contrib.rnn.GRUCell(self.params.hidden_dim)
			for t in range(self.params.memory_step):
				c = episode.update(memory, question, self.params.attention)
				if self.params.memory_update == 'gru':
					memory = gru(c, memory)[0]
				else:
					with tf.variable_scope(scope, reuse=False):
						memory = fully_connected(tf.concat([memory, c, question], 1), self.params.hidden_dim, 'MemoryUpdate',
						                         suffix=str(t), bn=False)

				h_q = fully_connected(tf.concat([memory, question], 1), self.params.hidden_dim, 'QuestionCoattention',
				                      activation='tanh')
				a_q = tf.nn.softmax(tf.reduce_sum(tf.transpose(questions, perm=[1, 0, 2]) * h_q, axis=2), dim=0)
				question = tf.transpose(tf.reduce_sum(tf.transpose(questions, perm=[2, 1, 0]) * a_q, axis=1))

				scope.reuse_variables()
		return memory

	def build_logits(self, memory, vocab_size, prefix):
		with tf.name_scope('Answer'):
			return fully_connected(memory, vocab_size, prefix, activation=None, bn=False)
