import build_vocabulary
import seq2seq_model
import modified_functions

import tensorflow as tf
import numpy as np

import sys
import collections
import time


def translate(l, dictionary, reversed_dictionary, add_tail=None):
	translated = list()
	if not l:
		return translated
	if isinstance(l[0], basestring):
		translated = [dictionary[word] if word in dictionary else build_vocabulary.UNK_ID for word in l]
	if isinstance(l[0], int):
		translated = [reversed_dictionary[word] for word in l]
	if add_tail is not None:
		translated.append(add_tail)
	return translated


dictionary, reversed_dictionary = build_vocabulary.dictionary, build_vocabulary.reversed_dictionary
data = build_vocabulary.bucket_data(seq2seq_model.buckets)
# data = build_vocabulary.bucket_data(seq2seq_model.buckets, build_vocabulary.toy_meta, build_vocabulary.toy_lines)
for bucket_id, qas in data.items():
	for i, (q, a) in enumerate(qas):
		data[bucket_id][i] = (
			translate(q, dictionary, reversed_dictionary),
			translate(a, dictionary, reversed_dictionary, add_tail=build_vocabulary.EOS_ID)
		)
buckets_size = [len(data[bucket_id]) for bucket_id in data]
buckets_scale = [float(bucket_size) / sum(buckets_size) for bucket_size in buckets_size]

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
		sentence = build_vocabulary.sentence2words(sentence)
		ids = translate(sentence, dictionary, reversed_dictionary)
		bucket_id = min([b for b in xrange(len(seq2seq_model.buckets)) if seq2seq_model.buckets[b][0] > len(ids)])

		encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(ids, [])]}, bucket_id)
		generation_results = model.step(sess, bucket_id, encoder_inputs, decoder_inputs, target_weights, True)
		outputs = [int(np.argmax(logit, axis=1)) for logit in generation_results[0]]
		if build_vocabulary.EOS_ID in outputs:
			outputs = outputs[:outputs.index(build_vocabulary.EOS_ID)]
		sentence = ' '.join(translate(outputs, dictionary, reversed_dictionary))
		sys.stdout.write(sentence)
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

