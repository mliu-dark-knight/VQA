from __future__ import print_function
from DMN import Base
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
	def __init__(self, hidden_dim, facts, attention, epsilon):
		self.hidden_dim = hidden_dim
		self.facts = tf.unstack(facts, axis=1)

		self.W1 = weight('EpisodeMemory_W1', [4 * hidden_dim, hidden_dim])
		self.b1 = bias('EpisodeMemory_b1', [hidden_dim])
		self.W2 = weight('EpisodeMemory_W2', [hidden_dim, 1])
		self.b2 = bias('EpisodeMemory_b2', [1])
		self.epsilon = epsilon
		if attention == 'gru':
			self.gru = AttentionGRU(self.hidden_dim, self.hidden_dim)

	def init_hidden(self, question):
		return tf.zeros_like(question)

	def update(self, memory, question, attention, images):
		gs = self.attention(self.facts, memory, question)

		tf.summary.image('visual attnmap', tf.concat([images, tf.image.resize_bicubic(tf.reshape(gs, [-1, 7, 7, 1]), [224, 224])], axis=3), max_outputs=16)
		if attention == 'soft':
			facts = tf.stack(self.facts, axis=1)
			return tf.reduce_sum(facts * gs, axis=1)

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
			g = tf.nn.softmax(tf.stack(Z, axis=1) / self.epsilon, dim=1)
			#g = tf.Print(g, [g], summarize=100)
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


class DMN(Base):
	def build(self):
		self.images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
		self.word2vec = embedding('Word2vec', shape=[self.params.vocab_size, self.params.glove_dim])
		self.input = tf.placeholder(tf.float32, shape=[None, self.params.img_size, self.params.channel_dim])
		self.question = tf.placeholder(tf.int32, shape=[None, self.params.max_ques_size])
		self.type = tf.placeholder(tf.int32, shape=[None])
		self.answer_b = tf.placeholder(tf.int32, shape=[None])
		self.answer_n = tf.placeholder(tf.int32, shape=[None])
		self.answer_m = tf.placeholder(tf.int32, shape=[None])
		self.answer_c = tf.placeholder(tf.int32, shape=[None])

		facts = self.build_input()
		questions = self.build_question(tf.nn.embedding_lookup(self.word2vec, self.question))
		type = self.build_type(tf.unstack(questions, axis=1)[-1])

		memory = self.build_memory(questions, facts)

		logits_b = self.build_logits(memory, 2, 'AnswerBinary', 'b')
		logits_n = self.build_logits(memory, self.params.num_range, 'AnswerNumber', 'n')
		logits_c = self.build_logits(memory, self.params.num_color, 'AnswerColor', 'c')
		logits_m = self.build_logits(memory, self.params.vocab_size, 'AnswerMulti', 'm')

		with tf.name_scope('Loss'):

			loss_t = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.type, logits=type))
			loss_m = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_m, logits=logits_m))
			loss_b = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_b, logits=logits_b))
			loss_n = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_n, logits=logits_n))
			loss_c = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_c, logits=logits_c))


			total_loss_m = loss_m + self.params.lambda_t * loss_t + self.params.lambda_r * tf.add_n(
				tf.get_collection('l2'))
			total_loss_b = loss_b + self.params.lambda_t * loss_t + self.params.lambda_r * tf.add_n(
				tf.get_collection('l2'))
			total_loss_n = loss_n + self.params.lambda_t * loss_t + self.params.lambda_r * tf.add_n(
				tf.get_collection('l2'))
			total_loss_c = loss_c + self.params.lambda_t * loss_t + self.params.lambda_r * tf.add_n(
				tf.get_collection('l2'))

		with tf.name_scope('Accuracy'):
			self.predicts_t = tf.cast(tf.argmax(type, 1), 'int32')
			self.predicts_b = tf.cast(tf.argmax(logits_b, 1), 'int32')
			self.predicts_n = tf.cast(tf.argmax(logits_n, 1), 'int32')
			self.predicts_m = tf.cast(tf.argmax(logits_m, 1), 'int32')
			self.predicts_c = tf.cast(tf.argmax(logits_c, 1), 'int32')
			self.accuracy_t = tf.reduce_mean(tf.cast(tf.equal(self.predicts_t, self.type), tf.float32))
			self.accuracy_b = tf.reduce_mean(tf.cast(tf.equal(self.predicts_b, self.answer_b), tf.float32))
			self.accuracy_n = tf.reduce_mean(tf.cast(tf.equal(self.predicts_n, self.answer_n), tf.float32))
			self.accuracy_m = tf.reduce_mean(tf.cast(tf.equal(self.predicts_m, self.answer_m), tf.float32))
			self.accuracy_c = tf.reduce_mean(tf.cast(tf.equal(self.predicts_c, self.answer_c), tf.float32))

		# learning_rate = tf.train.inverse_time_decay(self.params.learning_rate, self.global_step, 1, self.params.decay_rate)
		# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

		#debug_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DMN/Question_Embedding')
		# self.debug_b = optimizer.compute_gradients(total_loss_b, var_list=debug_var)
		# self.debug_m = optimizer.compute_gradients(total_loss_m, var_list=debug_var)

		self.gradient_descent_b = tf.train.AdamOptimizer(
			learning_rate=self.params.learning_rate).minimize(total_loss_b,
															  var_list=tf.get_collection('b') + tf.get_collection(
																  tf.GraphKeys.TRAINABLE_VARIABLES),
															  global_step=self.global_step,
															  colocate_gradients_with_ops=True)
		self.gradient_descent_n = tf.train.AdamOptimizer(
			learning_rate=self.params.learning_rate).minimize(total_loss_n,
															  var_list=tf.get_collection('n') + tf.get_collection(
																  tf.GraphKeys.TRAINABLE_VARIABLES),
															 # global_step=self.global_step,
															  colocate_gradients_with_ops=True)
		self.gradient_descent_m = tf.train.AdamOptimizer(
			learning_rate=self.params.learning_rate).minimize(total_loss_m,
															  var_list=tf.get_collection('m') + tf.get_collection(
																  tf.GraphKeys.TRAINABLE_VARIABLES),
															 # global_step=self.global_step,
															  colocate_gradients_with_ops=True)

		self.gradient_descent_c = tf.train.AdamOptimizer(
			learning_rate=self.params.learning_rate).minimize(total_loss_c,
															  var_list=tf.get_collection('c') + tf.get_collection(
																  tf.GraphKeys.TRAINABLE_VARIABLES),
															  # global_step=self.global_step,
															  colocate_gradients_with_ops=True)

		tf.summary.scalar('accuracy_t', self.accuracy_t)
		tf.summary.scalar('accuracy_b', self.accuracy_b, collections=['b_stuff'])
		tf.summary.scalar('accuracy_n', self.accuracy_n, collections=['n_stuff'])
		tf.summary.scalar("accuracy_m", self.accuracy_m, collections=['m_stuff'])
		tf.summary.scalar("accuracy_c", self.accuracy_c, collections=['c_stuff'])

		tf.summary.scalar('loss_t', loss_t)
		tf.summary.scalar('loss_b', loss_b, collections=['b_stuff'])
		tf.summary.scalar('loss_n', loss_n, collections=['n_stuff'])
		tf.summary.scalar('loss_m', loss_m, collections=['m_stuff'])
		tf.summary.scalar('loss_c', loss_c, collections=['c_stuff'])

		for variable in tf.trainable_variables():
			print(variable.name, variable.get_shape())

	def build_input(self):
		input = tf.reshape(self.input, [-1, self.params.channel_dim])
		facts = fully_connected(input, self.params.hidden_dim, 'ImageEmbedding', activation=None)
		return tf.reshape(facts, [-1, self.params.img_size, self.params.hidden_dim])

	def build_question(self, question):
		with tf.variable_scope('Question_Embedding'):
			if self.params.quasi_rnn:
				rnn_inputs = question
				for _ in range(self.params.rnn_layer):
					rnn = QuasiRNN(self.params.hidden_dim, self.params.kernel_width)
					rnn_inputs, _ = rnn(rnn_inputs, self.params.pooling)
				question_vecs = rnn_inputs
			else:
				gru = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.params.hidden_dim)] * self.params.rnn_layer)
				question_vecs, _ = tf.nn.dynamic_rnn(gru, question, dtype=tf.float32)
		return question_vecs

	def build_type(self, question):
		with tf.variable_scope('Question_Type'):
			return fully_connected(question, self.params.num_ques_type, 'Type', activation=None, bn=False)

	def build_memory(self, questions, facts):
		with tf.variable_scope('Memory') as scope:
			question = tf.identity(tf.unstack(questions, axis=1)[-1])
			episode = EpisodeMemory(self.params.hidden_dim, facts, self.params.attention, self.params.epsilon)
			memory = tf.identity(question)
			gru = tf.contrib.rnn.GRUCell(self.params.hidden_dim)
			for t in range(self.params.memory_step):
				c = episode.update(memory, question, self.params.attention, self.images)
				if self.params.memory_update == 'gru':
					memory = gru(c, memory)[0]
				else:
					with tf.variable_scope(scope, reuse=False):
						memory = fully_connected(tf.concat([memory, c, question], axis=1), self.params.hidden_dim, 'MemoryUpdate', suffix=str(t))

				h_q = fully_connected(tf.concat([memory, question], axis=1), self.params.hidden_dim, 'QuestionCoattention', activation='tanh')
				a_q = tf.nn.softmax(tf.reduce_sum(tf.transpose(questions, perm=[1, 0, 2]) * h_q, axis=2), dim=0)
				question = tf.transpose(tf.reduce_sum(tf.transpose(questions, perm=[2, 1, 0]) * a_q, axis=1))

				scope.reuse_variables()
		return memory

	def build_logits(self, memory, vocab_size, prefix, type):
		with tf.variable_scope('Answer'):
			return fully_connected(memory, vocab_size, prefix, activation=None, bn=False, type=type)
