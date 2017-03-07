from __future__ import print_function
from DMN import BaseModel
from NN import *


class AttentionGRU:
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
	def __init__(self, hidden_dim, question, facts):
		self.question = question
		self.hidden_dim = hidden_dim
		self.facts = tf.unstack(facts, axis=1)

		self.W1 = weight('EpisodeMemory_W1', [4 * hidden_dim, hidden_dim])
		self.b1 = bias('EpisodeMemory_b1', [hidden_dim])
		self.W2 = weight('EpisodeMemory_W2', [hidden_dim, 1])
		self.b2 = bias('EpisodeMemory_b2', [1])
		self.gru = AttentionGRU(self.hidden_dim, self.hidden_dim)

	def init_state(self):
		return tf.zeros_like(self.question)

	def update(self, memory):
		state = self.init_state()
		gs = self.attention(self.facts, memory)
		with tf.variable_scope('Attention_Gate') as scope:
			for f, g in zip(self.facts, gs):
				state = self.gru(f, state, g)
				scope.reuse_variables()
		return state

	def attention(self, fs, m):
		with tf.name_scope('Attention'):
			q = self.question
			Z = []
			for f in fs:
				z = tf.concat([f * q, f * m, tf.abs(f - q), tf.abs(f - m)], 1)
				Z.append(tf.matmul(tf.nn.tanh(tf.matmul(z, self.W1) + self.b1), self.W2) + self.b2)
			g = tf.nn.softmax(tf.stack(Z, axis=1))
		return tf.unstack(g, axis=1)



class DMN(BaseModel):
	def build(self):
		self.input = tf.placeholder(tf.float32, shape=[self.params.batch_size, self.params.img_size, self.params.channel_dim])
		self.question = tf.placeholder(tf.float32, shape=[self.params.batch_size, self.params.max_ques_size, self.params.glove_dim])
		self.answer = tf.placeholder(tf.int32, shape=[self.params.batch_size])

		facts = self.build_input()
		questions = self.build_question()
		memory = self.build_memory(questions, facts)
		logits = self.build_logits(memory)

		with tf.name_scope('Loss'):
			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer, logits=logits)
			loss = tf.reduce_mean(cross_entropy)
			total_loss = loss + tf.add_n(tf.get_collection('l2'))
			# tf.summary.scalar('Cross_Entropy', loss)

		with tf.name_scope('Accuracy'):
			self.predicts = tf.cast(tf.argmax(logits, 1), 'int32')
			corrects = tf.equal(self.predicts, self.answer)
			self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

		optimizer = tf.train.AdamOptimizer()
		self.gradient_descent = optimizer.minimize(total_loss, global_step=self.global_step)

		for variable in tf.trainable_variables():
			print(variable.name, variable.get_shape())

	def build_input(self):
		input_unpack = tf.unstack(self.input, axis=1)
		facts_unpack = []
		with tf.variable_scope('Input_Embedding') as scope:
			for f in input_unpack:
				facts_unpack.append(fully_connected(f, self.params.hidden_dim, 'FactsEmbedding', activation='tanh', bn=False))
				scope.reuse_variables()
		return tf.stack(facts_unpack, axis=1)

	def build_question(self):
		with tf.name_scope('Question_Embedding'):
			gru = tf.contrib.rnn.GRUCell(self.params.hidden_dim)
			question_vecs, _ = tf.nn.dynamic_rnn(gru, self.question, dtype=tf.float32)
		return question_vecs

	def build_memory(self, questions, facts):
		gru = tf.contrib.rnn.GRUCell(self.params.hidden_dim)
		with tf.variable_scope('Memory') as scope:
			question = tf.identity(tf.unstack(questions, axis=1)[-1])
			episode = EpisodeMemory(self.params.hidden_dim, question, facts)
			memory = tf.identity(question)
			for t in range(self.params.memory_step):
				if self.params.episode_memory:
					if self.params.memory_update == 'gru':
						memory = gru(episode.update(memory), memory)[0]
					else:
						c = episode.update(memory)
						with tf.variable_scope(scope, reuse=False):
							memory = fully_connected(tf.concat([memory, c, question], 1), self.params.hidden_dim, 'MemoryUpdate', suffix=str(t), bn=False)
				else:
					h_v = fully_connected(tf.concat([memory, question], 1), self.params.hidden_dim, 'MemoryUpdate', activation='tanh')
					a_v = tf.nn.softmax(tf.reduce_sum(tf.transpose(facts, perm=[1, 0, 2]) * h_v, axis=2), dim=0)
					memory = tf.transpose(tf.reduce_mean(tf.transpose(facts, perm=[2, 1, 0]) * a_v, axis=1))

				if self.params.question_coattention:
					h_q = fully_connected(tf.concat([memory, question], 1), self.params.hidden_dim, 'QuestionCoattention', activation='tanh')
					a_q = tf.nn.softmax(tf.reduce_sum(tf.transpose(questions, perm=[1, 0, 2]) * h_q, axis=2), dim=0)
					question = tf.transpose(tf.reduce_mean(tf.transpose(questions, perm=[2, 1, 0]) * a_q, axis=1))

				scope.reuse_variables()
		return memory

	def build_logits(self, memory):
		with tf.name_scope('Answer'):
			return fully_connected(memory, self.params.vocab_size, 'Answer', activation=None)
