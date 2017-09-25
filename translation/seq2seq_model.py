import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, legacy_seq2seq

import build_vocabulary


num_layers = 2
hidden_size = 256

buckets = [(5, 10), (10, 20), (20, 40), (30, 60)]
vocabulary_size_en = build_vocabulary.vocabulary_size_en
vacabulary_size_de = build_vocabulary.vocabulary_size_zh
num_samples = 512

max_gradient_norm = 5.0

PAD_ID = build_vocabulary.PAD_ID
GO_ID = build_vocabulary.GO_ID
EOS_ID = build_vocabulary.EOS_ID


class Seq2SeqDialogModel(object):

	def __init__(self, batch_size, learning_rate, learning_rate_decay):
		self.batch_size = batch_size
		self.encoder_inputs = list()
		self.decoder_inputs = list()
		self.target_weights = list()
		self.train_ops = list()
		self.losses_train = None
		self.outputs_train = None
		self.losses_feedpre = None
		self.outputs_feedpre = None

		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay)
		self.saver = None

	def setup_model(self):
		# sampled_softmax_loss function
		output_projection = None
		softmax_loss_function = None
		if num_samples < vacabulary_size_de:
			# w = tf.get_variable('proj_w', [hidden_size, vocabulary_size])
			w = tf.Variable(tf.truncated_normal([hidden_size, vacabulary_size_de], -0.1, 0.1))
			w_t = tf.transpose(w)
			# b = tf.get_variable('proj_b', [vocabulary_size])
			b = tf.Variable(tf.zeros([vacabulary_size_de]))
			output_projection = (w, b)

			def sampled_loss(labels, logits):
				labels = tf.reshape(labels, [-1, 1])
				return tf.nn.sampled_softmax_loss(
					weights=w_t, biases=b, labels=labels, inputs=logits,
					num_sampled=num_samples, num_classes=vacabulary_size_de)
			softmax_loss_function = sampled_loss

		# multi-layer rnn cell
		# cell = rnn.BasicLSTMCell(hidden_size)
		cell = rnn.GRUCell(hidden_size)
		if num_layers > 1:
			cell = rnn.MultiRNNCell([cell] * num_layers)

		# feeds
		for i in xrange(buckets[-1][0]):
			self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
		for i in xrange(buckets[-1][1] + 1):
			self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
			self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

		targets = [self.decoder_inputs[i+1] for i in xrange(len(self.decoder_inputs) - 1)]

		# seq2seq model structure
		def seq2seq_function(encoder_inputs, decoder_inputs, do_decode):
			return legacy_seq2seq.embedding_attention_seq2seq(
				encoder_inputs, decoder_inputs, cell,
				num_encoder_symbols=vocabulary_size_en,
				num_decoder_symbols=vacabulary_size_de,
				embedding_size=hidden_size,
				output_projection=output_projection,
				feed_previous=do_decode
			)

		with tf.variable_scope("reusable_model"):
			self.outputs_train, self.losses_train = legacy_seq2seq.model_with_buckets(
				self.encoder_inputs, self.decoder_inputs, targets, self.target_weights, buckets,
				lambda x, y: seq2seq_function(x, y, False),
				softmax_loss_function=softmax_loss_function
				)

		with tf.variable_scope("reusable_model", reuse=True):
			self.outputs_feedpre, self.losses_feedpre = legacy_seq2seq.model_with_buckets(
				self.encoder_inputs, self.decoder_inputs, targets, self.target_weights, buckets,
				lambda x, y: seq2seq_function(x, y, True),
				softmax_loss_function=softmax_loss_function
				)
			if output_projection is not None:
				for b in xrange(len(buckets)):
					self.outputs_feedpre[b] = [
						tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs_feedpre[b]
					]

		# train op
		params = tf.trainable_variables()
		for b in xrange(len(buckets)):
			gradients = tf.gradients(self.losses_train[b], params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
			self.train_ops.append(tf.train.GradientDescentOptimizer(self.learning_rate).apply_gradients(
				zip(clipped_gradients, params)
			))

		# saver
		self.saver = tf.train.Saver(max_to_keep=1)

	def step(self, sess, bucket_id, encoder_inputs, decoder_inputs, target_weights, forward_only):
		encoder_size, decoder_size = buckets[bucket_id]
		assert len(encoder_inputs) == encoder_size
		assert len(decoder_inputs) == decoder_size
		assert len(target_weights) == decoder_size

		# input feed
		input_feed = dict()
		for l in xrange(encoder_size):
			input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
		for l in xrange(decoder_size):
			input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
			input_feed[self.target_weights[l].name] = target_weights[l]
		input_feed[self.decoder_inputs[decoder_size].name] = np.zeros([self.batch_size], dtype=np.int32)

		# outputs
		if not forward_only:
			outputs = sess.run([self.train_ops[bucket_id], self.losses_train[bucket_id]], input_feed)
		else:
			outputs = sess.run([self.outputs_feedpre[bucket_id], self.losses_feedpre[bucket_id]], input_feed)
		return outputs

	def get_batch(self, data, bucket_id):
		encoder_size, decoder_size = buckets[bucket_id]
		encoder_inputs = list()   # batch_size * [encoder_size]
		decoder_inputs = list()   # batch_size * [decoder_size]

		# choose in data[bucket_id]
		for _ in xrange(self.batch_size):
			encoder_input, decoder_input = random.choice(data[bucket_id])
			encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
			encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

			decoder_pad_size = decoder_size - len(decoder_input) - 1
			decoder_inputs.append([GO_ID] + decoder_input + [PAD_ID] * decoder_pad_size)

		# reindex
		batch_encoder_inputs = list()   # encoder_size * [batch_size]
		batch_decoder_inputs = list()   # decoder_size * [batch_size]
		batch_weights = list()
		for i in xrange(encoder_size):
			batch_encoder_inputs.append(
				np.array([encoder_inputs[j][i] for j in xrange(self.batch_size)], dtype=np.int32)
			)
		for i in xrange(decoder_size):
			batch_decoder_inputs.append(
				np.array([decoder_inputs[j][i] for j in xrange(self.batch_size)], dtype=np.int32)
			)
			# target weights
			batch_weight = np.ones(self.batch_size, dtype=np.float32)
			for batch in xrange(self.batch_size):
				if i + 1 < decoder_size:
					target = decoder_inputs[batch][i + 1]
				if i + 1 == decoder_size or target == PAD_ID:
					batch_weight[batch] = 0.0
			batch_weights.append(batch_weight)

		return batch_encoder_inputs, batch_decoder_inputs, batch_weights

