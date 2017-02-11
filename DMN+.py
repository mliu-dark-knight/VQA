import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from NN import *



class AttentionGRU:
    def __init__(self, num_units, input_dim):
        self.num_units = num_units
        self.input_dim = input_dim

    def __call__(self, inputs, state, attention, scope=None):
        with tf.variable_scope(scope or 'AttentionGRU'):
            r = tf.nn.sigmoid(self._linear(inputs, state, bias_default=1.0))
            h = tanh(self.linear(inputs, r * state))
            new_h = attention * h + (1 - attention) * state
        return new_h

    def linear(self, x, h, bias_default=0.0):
        W = weight('W', [self.num_units, self.input_dim])
        U = weight('U', [self.num_units, self.num_units])
        b = bias('b', self.num_units, bias_default)
        return tf.matmul(W, x) + tf.matmul(U, h) + b



class EpisodeModule:
    """ Inner GRU module in episodic memory that creates episode vector. """
    def __init__(self, num_hidden, question, facts, is_training, bn):
        self.question = question
        self.facts = tf.unpack(tf.transpose(facts, [1, 2, 0]))  # F x [d, N]

        # transposing for attention
        self.question_transposed = tf.transpose(question)
        self.facts_transposed = [tf.transpose(f) for f in self.facts]  # F x [N, d]

        # parameters
        self.w1 = weight('w1', [num_hidden, 4 * num_hidden])
        self.b1 = bias('b1', [num_hidden, 1])
        self.w2 = weight('w2', [1, num_hidden])
        self.b2 = bias('b2', [1, 1])
        self.gru = AttnGRU(num_hidden, is_training, bn)

    @property
    def init_state(self):
        return tf.zeros_like(self.facts_transposed[0])

    def new(self, memory):
        """ Creates new episode vector (will feed into Episodic Memory GRU)
        :param memory: Previous memory vector
        :return: episode vector
        """
        state = self.init_state
        memory = tf.transpose(memory)  # [N, D]

        with tf.variable_scope('AttnGate') as scope:
            for f, f_t in zip(self.facts, self.facts_transposed):
                g = self.attention(f, memory)
                state = self.gru(f_t, state, g)
                scope.reuse_variables()  # share params

        return state

    def attention(self, f, m):
        """ Attention mechanism. For details, see paper.
        :param f: A fact vector [N, D] at timestep
        :param m: Previous memory vector [N, D]
        :return: attention vector at timestep
        """
        with tf.variable_scope('attention'):
            # NOTE THAT instead of L1 norm we used L2
            q = self.question_transposed
            vec = tf.concat(0, [f * q, f * m, tf.abs(f - q), tf.abs(f - m)])  # [4*d, N]

            # attention learning
            l1 = tf.matmul(self.w1, vec) + self.b1  # [N, d]
            l1 = tf.nn.tanh(l1)
            l2 = tf.matmul(self.w2, l1) + self.b2
            l2 = tf.nn.softmax(l2)
            return tf.transpose(l2)

        return att



class DMN(BaseModel):
    """ Dynamic Memory Networks (March 2016 Version - https://arxiv.org/abs/1603.01417)
        Improved End-To-End version."""
    def build(self):
        params = self.params
        N, L, Q, F = params.batch_size, params.max_sent_size, params.max_ques_size, params.max_fact_count
        V, d, A = params.embed_size, params.hidden_size, self.words.vocab_size

        # initialize self
        # placeholders
        input = tf.placeholder('int32', shape=[N, F, L], name='x')  # [num_batch, fact_count, sentence_len]
        question = tf.placeholder('int32', shape=[N, Q], name='q')  # [num_batch, question_len]
        answer = tf.placeholder('int32', shape=[N], name='y')  # [num_batch] - one word answer
        fact_counts = tf.placeholder('int64', shape=[N], name='fc')
        input_mask = tf.placeholder('float32', shape=[N, F, L, V], name='xm')
        is_training = tf.placeholder(tf.bool)
        self.att = tf.constant(0.)

        # Prepare parameters
        gru = rnn_cell.GRUCell(d)
        l = self.positional_encoding()
        embedding = weight('embedding', [A, V], init='uniform', range=3**(1/2))

        # dropout time
        facts = dropout(facts, params.keep_prob, is_training)

        with tf.name_scope('InputFusion'):
            # Bidirectional RNN
            with tf.variable_scope('Forward'):
                forward_states, _ = tf.nn.dynamic_rnn(gru, facts, fact_counts, dtype=tf.float32)

            with tf.variable_scope('Backward'):
                facts_reverse = tf.reverse_sequence(facts, fact_counts, 1)
                backward_states, _ = tf.nn.dynamic_rnn(gru, facts_reverse, fact_counts, dtype=tf.float32)

            # Use forward and backward states both
            facts = forward_states + backward_states  # [N, F, d]

        with tf.variable_scope('Question'):
            ques_list = tf.unpack(tf.transpose(question))
            ques_embed = [tf.nn.embedding_lookup(embedding, w) for w in ques_list]
            _, question_vec = tf.nn.rnn(gru, ques_embed, dtype=tf.float32)

        # Episodic Memory
        with tf.variable_scope('Episodic'):
            episode = EpisodeModule(d, question_vec, facts, is_training, params.batch_norm)
            memory = tf.identity(question_vec)

            for t in range(params.memory_step):
                with tf.variable_scope('Layer%d' % t) as scope:
                    if params.memory_update == 'gru':
                        memory = gru(episode.new(memory), memory)[0]
                    else:
                        # ReLU update
                        c = episode.new(memory)
                        concated = tf.concat(1, [memory, c, question_vec])

                        w_t = weight('w_t', [3 * d, d])
                        z = tf.matmul(concated, w_t)
                        if params.batch_norm:
                            z = batch_norm(z, is_training)
                        else:
                            b_t = bias('b_t', d)
                            z = z + b_t
                        memory = tf.nn.relu(z)  # [N, d]

                    scope.reuse_variables()

        # Regularizations
        if params.batch_norm:
            memory = batch_norm(memory, is_training=is_training)
        memory = dropout(memory, params.keep_prob, is_training)

        with tf.name_scope('Answer'):
            # Answer module : feed-forward version (for it is one word answer)
            w_a = weight('w_a', [d, A], init='xavier')
            logits = tf.matmul(memory, w_a)  # [N, A]

        with tf.name_scope('Loss'):
            # Cross-Entropy loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, answer)
            loss = tf.reduce_mean(cross_entropy)
            total_loss = loss + params.weight_decay * tf.add_n(tf.get_collection('l2'))

        with tf.variable_scope('Accuracy'):
            # Accuracy
            predicts = tf.cast(tf.argmax(logits, 1), 'int32')
            corrects = tf.equal(predicts, answer)
            num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
            accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        # Training
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        opt_op = optimizer.minimize(total_loss, global_step=self.global_step)

        # placeholders
        self.x = input
        self.xm = input_mask
        self.q = question
        self.y = answer
        self.fc = fact_counts
        self.is_training = is_training

        # tensors
        self.total_loss = total_loss
        self.num_corrects = num_corrects
        self.accuracy = accuracy
        self.opt_op = opt_op

    def preprocess_batch(self, batches):
        """ Make padding and masks last word of sentence. (EOS token)
        :param batches: A tuple (input, question, label, mask)
        :return A tuple (input, question, label, mask)
        """
        params = self.params
        input, question, label = batches
        N, L, Q, F = params.batch_size, params.max_sent_size, params.max_ques_size, params.max_fact_count
        V = params.embed_size

        # make input and question fixed size
        new_input = np.zeros([N, F, L])  # zero padding
        input_masks = np.zeros([N, F, L, V])
        new_question = np.zeros([N, Q])
        new_labels = []
        fact_counts = []

        for n in range(N):
            for i, sentence in enumerate(input[n]):
                sentence_len = len(sentence)
                new_input[n, i, :sentence_len] = [self.words.word_to_index(w) for w in sentence]
                input_masks[n, i, :sentence_len, :] = 1.  # mask words

            fact_counts.append(len(input[n]))

            sentence_len = len(question[n])
            new_question[n, :sentence_len] = [self.words.word_to_index(w) for w in question[n]]

            new_labels.append(self.words.word_to_index(label[n]))

        return new_input, new_question, new_labels, fact_counts, input_masks

    def get_feed_dict(self, batches, is_train):
        input, question, label, fact_counts, mask = self.preprocess_batch(batches)
        return {
            self.x: input,
            self.xm: mask,
            self.q: question,
            self.y: label,
            self.fc: fact_counts,
            self.is_training: is_train
        }


class DMN(BaseModel):
    """ Dynamic Memory Networks (http://arxiv.org/abs/1506.07285)
        Semantic Memory version: Instead of implementing embedding layer,
        it uses GloVe instead. (First version of DMN paper.)
    """
    def build(self):
        params = self.params
        N, Q, F = params.batch_size, params.max_ques_size, params.img_size
        V, d, A = params.glove_size, params.hidden_size, self.words.vocab_size

        input = tf.placeholder(tf.float32, shape=[self.params.batch_size, self.params.img_size, self.params.channel_dim], name='x')
        question = tf.placeholder(tf.float32, shape=[self.params.batch_size, self.params.max_ques_size, self.params.glove_dim], name='q')
        answer = tf.placeholder(tf.int64, shape=[self.params.batch_size], name='y')

        is_training = tf.placeholder(tf.bool)

        # Prepare parameters
        gru = rnn_cell.GRUCell(self.params.hidden_dim)

        facts = fully_connected(input, self.params.hidden_dim, activation='tanh', bn=False)

        question_vec = self.build_question()
        memory = tf.identity(question_vec)
        self.build_memory(memory, facts)
        self.build_answer()

        # Regularizations
        if params.batch_norm:
            memory = batch_norm(memory, is_training=is_training)
        memory = dropout(memory, params.keep_prob, is_training)

        with tf.name_scope('Loss'):
            # Cross-Entropy loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, answer)
            loss = tf.reduce_mean(cross_entropy)
            total_loss = loss + tf.add_n(tf.get_collection('l2'))

        with tf.variable_scope('Accuracy'):
            # Accuracy
            predicts = tf.cast(tf.argmax(logits, 1), 'int32')
            corrects = tf.equal(predicts, answer)
            num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
            accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        # Training
        optimizer = tf.train.AdadeltaOptimizer(params.learning_rate)
        opt_op = optimizer.minimize(total_loss, global_step=self.global_step)

        # placeholders
        self.x = input
        self.q = question
        self.y = answer
        self.mask = input_mask
        self.is_training = is_training

        # tensors
        self.total_loss = total_loss
        self.num_corrects = num_corrects
        self.accuracy = accuracy
        self.opt_op = opt_op

    def build_question(self):
        with tf.variable_scope('Question') as scope:
            gru = rnn_cell.GRUCell(self.params.hidden_size)
            question_vecs, _ = rnn.rnn(gru, questions)
            question_vec = question_vecs[-1]
        return question_vec


    def build_answer(self):
        with tf.name_scope('Answer'):
            w_a = weight('w_a', [d, A])
            self.logits = tf.matmul(memory, w_a)

    def build_memory(self, question_vec, facts):
        with tf.variable_scope('Memory') as scope:
            episode = EpisodeModule(self.params.hidden_dim, question_vec, facts)

            memory = tf.identity(question_vec)
            for t in range(params.memory_step):
                memory = gru(episode.new(memory), memory)[0]
                scope.reuse_variables()



    def make_decoder_batch_input(self, input):
        """ Reshape batch data to be compatible with Seq2Seq RNN decoder.
        :param input: Input 3D tensor that has shape [num_batch, sentence_len, wordvec_dim]
        :return: list of 2D tensor that has shape [num_batch, wordvec_dim]
        """
        input_transposed = tf.transpose(input, [1, 0, 2])  # [L, N, V]
        return tf.unpack(input_transposed)

    def preprocess_batch(self, batches):
        """ Vectorizes padding and masks last word of sentence. (EOS token)
        :param batches: A tuple (input, question, label, mask)
        :return A tuple (input, question, label, mask)
        """
        params = self.params
        input, question, label = batches
        N, Q, F, V = params.batch_size, params.max_ques_size, params.max_fact_count, params.embed_size

        # calculate max sentence size
        L = 0
        for n in range(N):
            sent_len = np.sum([len(sentence) for sentence in input[n]])
            L = max(L, sent_len)
        params.max_sent_size = L

        # make input and question fixed size
        new_input = np.zeros([N, L, V])  # zero padding
        new_question = np.zeros([N, Q, V])
        new_mask = np.full([N, L], False, dtype=bool)
        new_labels = []

        for n in range(N):
            sentence = np.array(input[n]).flatten()  # concat all sentences
            sentence_len = len(sentence)

            input_mask = [index for index, w in enumerate(sentence) if w == '.']
            new_input[n, :sentence_len] = [self.words.vectorize(w) for w in sentence]

            sentence_len = len(question[n])
            new_question[n, :sentence_len] = [self.words.vectorize(w) for w in question[n]]

            new_labels.append(self.words.word_to_index(label[n]))

            # mask on
            for eos_index in input_mask:
                new_mask[n, eos_index] = True

        return new_input, new_question, new_labels, new_mask

    def get_feed_dict(self, batches, is_train):
        input, question, label, mask = self.preprocess_batch(batches)
        return {
            self.x: input,
            self.q: question,
            self.y: label,
            self.mask: mask,
            self.is_training: is_train
        }

