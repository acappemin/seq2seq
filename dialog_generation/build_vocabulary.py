import re
import collections
import movie_corpus


PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

vocabulary_size = 20000

_WORD_SPILT = re.compile(r"[.,!?\"':;)(]")
_DIGIT_RE = re.compile(r"\d+")

meta, lines = movie_corpus.meta, movie_corpus.lines


def sentence2words(sentence):
	words = list()
	for space_separated_fragment in sentence.strip().split():
		words_candidate = re.split(_WORD_SPILT, space_separated_fragment)
		words.extend([re.sub(_DIGIT_RE, r'0', w.lower()) for w in words_candidate if w])
	return words


def build_vocabulary(lines):
	words = list()
	for key, sentence in lines.items():
		new_sentence = sentence2words(sentence)
		words.extend(new_sentence)
		lines[key] = new_sentence

	count = [['__UNK_', 0]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 4))

	dictionary = {
		'__PAD_': PAD_ID,
		'__GO_': GO_ID,
		'__EOS_': EOS_ID
	}
	for word, _ in count:
		dictionary[word] = len(dictionary)

	unk_count = 0
	for word in words:
		index = dictionary.get(word, UNK_ID)
		if index == UNK_ID:
			unk_count += 1
	count[0][1] = unk_count

	reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

	# dictionary: {'__PAD_': 0, '__GO_': 1, '__EOS_': 2, '__UNK_': 3, ...}
	# reversed_dictionary: {int: str}
	return dictionary, reversed_dictionary, lines

dictionary, reversed_dictionary, separated_lines = build_vocabulary(lines)


def bucket_data(buckets, meta=meta, lines=separated_lines):
	data = dict()
	for bucket_id in xrange(len(buckets)):
		data[bucket_id] = list()
	for q, a in meta:
		q = lines[q]
		a = lines[a]
		for bucket_id in xrange(len(buckets)):
			# GO in the head & EOS in the tail
			# decoder_size = bucket_size + 1, len(a) + 1 <= bucket_size
			if len(q) <= buckets[bucket_id][0] and len(a) + 1 <= buckets[bucket_id][1]:
				data[bucket_id].append((q, a))
				break
	return data


toy_meta = [
	(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)
]
toy_lines = {
	1: 'hi', 2: 'hi',
	3: 'hello', 4: 'hello',
	5: 'how are you', 6: 'i am fine',
	7: 'what is your name', 8: 'jack',
	9: 'thank you', 10: 'you are welcome',
}
for key, value in toy_lines.items():
	toy_lines[key] = sentence2words(value)

