import os
import urllib
import zipfile


url = 'http://www.mpi-sws.org/~cristian/data/'


def maybe_download(filename, expected_bytes):
	"""Download a file if not present, and make sure it's the right size."""
	local_filename = os.path.join(os.getcwd(), filename)
	if not os.path.exists(local_filename):
		local_filename, _ = urllib.urlretrieve(url + filename, local_filename)
	statinfo = os.stat(local_filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified', filename)
	else:
		print(statinfo.st_size)
		raise Exception('Failed to verify ' + local_filename + '. Can you get to it with a browser?')
	return local_filename

filename = maybe_download('cornell_movie_dialogs_corpus.zip', 9916637)


def generate_meta(meta_data):
	qa_pairs = []
	for i, conversation in enumerate(meta_data):
		index = conversation.find('[')
		if index != -1:
			conversation = eval(conversation[index:])
			for j in range(0, len(conversation)-1, 2):
				qa_pairs.append((conversation[j], conversation[j+1]))
	return qa_pairs


def generate_lines(lines_data):
	lines_dict = {}
	for i, line in enumerate(lines_data):
		values = line.split(' +++$+++ ')
		if len(values) == 5:
			lines_dict[values[0]] = values[-1]
	return lines_dict


def read_data(filename):
	"""Extract the first file enclosed in a zip file as a list of words."""
	with zipfile.ZipFile(filename) as f:
		meta_data = f.read('cornell movie-dialogs corpus/movie_conversations.txt').split('\n')
		meta = generate_meta(meta_data)
		lines_data = f.read('cornell movie-dialogs corpus/movie_lines.txt').split('\n')
		lines = generate_lines(lines_data)
	return meta, lines

meta, lines = read_data(filename)
data_len = len(meta)
print('Data size', data_len)

