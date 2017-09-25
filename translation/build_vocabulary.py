# -*- encoding:utf-8 -*-

import re
import collections


tar_filename = 'training-parallel-nc-v12.tgz'
filenames = {
	'en': 'training/news-commentary-v12.zh-en.en',
	'zh': 'training/news-commentary-v12.zh-en.zh'
}

# import tarfile
# tar = tarfile.open(tar_filename, 'r')
# for filename in filenames.values():
# 	tar.extract(filename)

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

vocabulary_size_en = 20000
vocabulary_size_zh = 4000

_WORD_SPILT_EN = ['.', ',', '!', '?', '"', '\'', ':', ';', '(', ')']
_WORD_SPILT_ZH = ['。', '，', '！', '？', '“', '”', '‘', '’', '：', '；', '（', '）']
_WORD_SPILT_U = [splitter.decode('utf8') for splitter in _WORD_SPILT_ZH + _WORD_SPILT_EN]
_WORD_SPILT = re.compile(r"[.,!?\"':;)(]")
_DIGIT_RE = re.compile(r"\d+")


def sentence2words_en(sentence):
	words = list()
	for space_separated_fragment in sentence.strip().split():
		words_candidate = re.split(_WORD_SPILT, space_separated_fragment)
		words.extend([re.sub(_DIGIT_RE, '0', w.lower()) for w in words_candidate if w])
	return words


def sentence2words_zh(sentence):
	words = list()
	sentence = re.sub(_DIGIT_RE, '0', sentence.strip())
	for char in sentence:
		if char not in _WORD_SPILT_U:
			words.append(char)
	return words


def build_vocabulary(lines, vocabulary_size, sentence2words):
	words = list()
	for index in xrange(len(lines)):
		new_sentence = sentence2words(lines[index])
		words.extend(new_sentence)
		lines[index] = new_sentence

	all_count = collections.Counter(words)
	print 'length of all vocabulary: ', len(all_count)
	assert len(all_count) >= vocabulary_size
	count = [['__UNK_', 0]]
	count.extend(all_count.most_common(vocabulary_size - 4))

	# for i, char in enumerate(count[:10]):
	# 	print char[0].encode('utf8')

	dictionary = {
		'__PAD_': PAD_ID,
		'__GO_': GO_ID,
		'__EOS_': EOS_ID,
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


def translate(l, dictionary, reversed_dictionary, add_tail=None):
	translated = list()
	if not l:
		return translated
	if isinstance(l[0], basestring):
		translated = [dictionary[word] if word in dictionary else UNK_ID for word in l]
	if isinstance(l[0], int):
		translated = [reversed_dictionary[word] for word in l]
	if add_tail is not None:
		translated.append(add_tail)
	return translated


def read_file():
	with open(filenames['en'], 'rb') as f:
		lines = f.readlines()
		dictionary_en, reversed_dictionary_en, lines_en = build_vocabulary(lines, vocabulary_size_en, sentence2words_en)
		digit_lines_en = [translate(line_en, dictionary_en, reversed_dictionary_en) for line_en in lines_en]

	with open(filenames['zh'], 'rb') as f:
		lines = f.readlines()
		lines = [line.decode('utf8') for line in lines]
		dictionary_zh, reversed_dictionary_zh, lines_zh = build_vocabulary(lines, vocabulary_size_zh, sentence2words_zh)
		digit_lines_zh = [
			translate(line_zh, dictionary_zh, reversed_dictionary_zh, add_tail=EOS_ID) for line_zh in lines_zh
			]

	assert len(lines_en) == len(lines_zh)
	print 'len_data: %d' % len(lines_en)

	return dictionary_en, reversed_dictionary_en, lines_en, digit_lines_en,\
		dictionary_zh, reversed_dictionary_zh, lines_zh, digit_lines_zh

dictionary_en, reversed_dictionary_en, lines_en, digit_lines_en,\
dictionary_zh, reversed_dictionary_zh, lines_zh, digit_lines_zh = read_file()

ls_en = [len(line) for line in lines_en]
ls_zh = [len(line) for line in lines_zh]
print 'avg_len_en: %f, max_len_en: %d' % (sum(ls_en) / float(len(ls_en)), max(ls_en))
print 'avg_len_zh: %f, max_len_zh: %d' % (sum(ls_zh) / float(len(ls_zh)), max(ls_zh))

# index = 1
# print lines_en[index]
# for word in lines_zh[index]:
# 	print word.encode('utf8')


def bucket_data(buckets, lines_en_local=digit_lines_en, lines_zh_local=digit_lines_zh):
	assert len(lines_en_local) == len(lines_zh_local)
	data = dict()
	for bucket_id in xrange(len(buckets)):
		data[bucket_id] = list()
	for line_en, line_zh in zip(lines_en_local, lines_zh_local):
		for bucket_id in xrange(len(buckets)):
			# EOS already in the tail
			# decoder_size = bucket_size + 1, len(a) <= bucket_size
			if len(line_en) <= buckets[bucket_id][0] and len(line_zh) <= buckets[bucket_id][1]:
				data[bucket_id].append((line_en, line_zh))
				break
	# save memory
	global lines_en, digit_lines_en, lines_zh, digit_lines_zh
	del lines_en
	del digit_lines_en
	del lines_zh
	del digit_lines_zh
	return data


toy_lines_en = [
	['you', 'are', 'welcome'],
	['what', 'is', 'your', 'name'],
	['thank', 'you'],
	['good', 'night'],
	['see', 'you']
]
toy_lines_zh = [
	[u'不', u'客', u'气'],
	[u'你', u'叫', u'什', u'么', u'名', u'字'],
	[u'谢', u'谢'],
	[u'晚', u'安'],
	[u'再', u'见']
]
toy_digit_lines_en = [translate(line, dictionary_en, reversed_dictionary_en) for line in toy_lines_en]
toy_digit_lines_zh = [translate(line, dictionary_zh, reversed_dictionary_zh, add_tail=EOS_ID) for line in toy_lines_zh]
# print toy_digit_lines_en
# print toy_digit_lines_zh

