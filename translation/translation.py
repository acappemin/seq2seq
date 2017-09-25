# -*- encoding:utf-8 -*-

import build_vocabulary
import seq2seq_model
import modified_functions

import tensorflow as tf
import numpy as np

import sys
import collections
import time


dictionary_en, reversed_dictionary_en = build_vocabulary.dictionary_en, build_vocabulary.reversed_dictionary_en
dictionary_zh, reversed_dictionary_zh = build_vocabulary.dictionary_zh, build_vocabulary.reversed_dictionary_zh
data = build_vocabulary.bucket_data(seq2seq_model.buckets)
# data = build_vocabulary.bucket_data(
# 	seq2seq_model.buckets, build_vocabulary.toy_digit_lines_en, build_vocabulary.toy_digit_lines_zh)

buckets_size = [len(data[bucket_id]) for bucket_id in data]
buckets_scale = [float(bucket_size) / sum(buckets_size) for bucket_size in buckets_size]
print 'buckets_size: %s' % buckets_size
# scale adjustment
buckets_scale = [1.0 / len(buckets_scale) for _ in buckets_scale]

batch_size = 64
learning_rate = 0.5
learning_rate_decay = 0.95
model = seq2seq_model.Seq2SeqDialogModel(
	batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay)
model.setup_model()


def train(saver_path='saver/model.ckpt'):
	num_steps = 20000
	check_steps = 50
	save_steps = 99999999

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		losses = collections.deque(maxlen=3)
		avg_loss = 0.0
		while True:
			model.batch_size = batch_size
			time1 = time.time()
			for step in xrange(num_steps):
				random_number_01 = np.random.random_sample()
				bucket_id = min([i for i in xrange(len(buckets_scale)) if sum(buckets_scale[:i+1]) > random_number_01])
				encoder_inputs, decoder_inputs, target_weights = model.get_batch(data, bucket_id)
				train_results = model.step(sess, bucket_id, encoder_inputs, decoder_inputs, target_weights, False)
				avg_loss += train_results[1] / check_steps

				if (step + 1) % check_steps == 0:
					time2 = time.time()
					delta_time = (time2 - time1) / 60.0
					time1 = time2
					print 'Step %d, Loss %f, Time %smin' % (step, avg_loss, delta_time)
					if len(losses) > 2 and avg_loss > max(losses):
						_, learning_rate = sess.run([model.learning_rate_decay_op, model.learning_rate])
						print 'Learning Rate Decay To %f' % learning_rate
					losses.append(avg_loss)
					avg_loss = 0.0

				if (step + 1) % save_steps == 0:
					model.saver.save(sess, saver_path, global_step=step)

			do_test(sess)


def test(saver_path='saver/model.ckpt'):
	with tf.Session() as sess:
		model.saver.restore(sess, saver_path)
		do_test(sess)


def do_test(sess):
	model.batch_size = 1
	sys.stdout.write('>')
	sys.stdout.flush()
	while True:
		sentence = sys.stdin.readline()
		sentence = sentence.strip()
		if sentence == 'break':
			break
		sentence = build_vocabulary.sentence2words_en(sentence)
		ids = build_vocabulary.translate(sentence, dictionary_en, reversed_dictionary_en)
		if build_vocabulary.UNK_ID in ids:
			sys.stdout.write('has unknown words\n')
			sys.stdout.flush()
			continue

		bucket_id = min([b for b in xrange(len(seq2seq_model.buckets)) if seq2seq_model.buckets[b][0] > len(ids)])
		encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(ids, [])]}, bucket_id)
		generation_results = model.step(sess, bucket_id, encoder_inputs, decoder_inputs, target_weights, True)
		outputs = [int(np.argmax(logit, axis=1)) for logit in generation_results[0]]
		if build_vocabulary.EOS_ID in outputs:
			outputs = outputs[:outputs.index(build_vocabulary.EOS_ID)]
		sentence = ''.join(build_vocabulary.translate(outputs, dictionary_zh, reversed_dictionary_zh))
		sys.stdout.write(sentence.encode('utf8'))
		sys.stdout.write('\n')
		sys.stdout.flush()


if __name__ == '__main__':
	# train or test
	print 'Choose Operation: train/test?'
	while True:
		operation = sys.stdin.readline()
		operation = operation.strip()
		if operation == 'train':
			train()
		if operation == 'test':
			test()

