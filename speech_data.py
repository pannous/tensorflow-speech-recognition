#!/usr/bin/env python
#!/usr/local/bin/python
"""Utilities for downloading and providing data from openslr.org, libriSpeech, Pannous, Gutenberg, WMT, tokenizing, vocabularies."""
# TODO! see https://github.com/pannous/caffe-speech-recognition for some data sources

import os
import re
import sys
import wave

import numpy
import numpy as np
import skimage.io  # scikit-image

try:
	import librosa
except:
	print("pip install librosa ; if you want mfcc_batch_generator")
# import extensions as xx
from random import shuffle
try:
	from six.moves import urllib
	from six.moves import xrange  # pylint: disable=redefined-builtin
except:
	pass # fuck 2to3

speech_commands="sheila seven right one house down zero go yes wow six no three happy \
bird stop marvin two five on off four dog up tree cat bed nine eight left".split(" ")

# TRAIN_INDEX='train_words_index.txt'
# TEST_INDEX='test_words_index.txt'
SOURCE_URL = 'http://pannous.net/files/' #spoken_numbers.tar'

DATA_DIR = 'data/'
pcm_path = "data/spoken_numbers_pcm/" # 8 bit
wav_path = "data/spoken_numbers_wav/" # 16 bit s16le
path = pcm_path
CHUNK = 4096
test_fraction=0.1 # 10% of data for test / verification

class Source:  # labels
	DIGIT_WAVES = './spoken_numbers_pcm.tar'
	# DIGIT_SPECTROS = 'spoken_numbers_spectros_64x64.tar'  # 64x64  baby data set, works astonishingly well
	DIGIT_SPECTROS = './spoken_numbers_spectros_64x64.tar'
	NUMBER_WAVES = 'spoken_numbers_wav.tar'
	NUMBER_IMAGES = 'spoken_numbers.tar'  # width=256 height=256
	WORD_SPECTROS = 'https://dl.dropboxusercontent.com/u/23615316/spoken_words.tar'  # width,height=512# todo: sliding window!
	SPEECH_COMMANDS = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz" # 16000Hz 1ch s16le (2 bytes per sample)
	WORD_WAVES = 'spoken_words_wav.tar'
	TEST_INDEX = 'test_index.txt'
	TRAIN_INDEX = 'train_index.txt'

from enum import Enum
class Target(Enum):  # labels
	digits=1
	speaker=2
	words_per_minute=3
	word_phonemes=4
	word = 5  # int vector as opposed to binary hotword
	sentence=6
	sentiment=7
	first_letter=8
	hotword = 9
	speech_commands = 10
	# test_word=9 # use 5 even for speaker etc


num_characters = 32
# num_characters=60 #  only one case, Including numbers
# num_characters=128 #
# num_characters=256 #  including special characters
# offset=0  # 1:1 mapping ++
# offset=32 # starting with ' ' space
# offset=48 # starting with  numbers
offset = 64  # starting with characters
max_word_length = 20
terminal_symbol = 0

def pad(vec, pad_to=max_word_length, one_hot=False,paddy=terminal_symbol):
	for i in range(0, pad_to - len(vec)):
		if one_hot:
			vec.append([paddy] * num_characters)
		else:
			vec.append(paddy)
	return vec

def char_to_class(c):
	return (ord(c) - offset) % num_characters

def string_to_int_word(word, pad_to):
	z = map(char_to_class, word)
	z = list(z)
	z = pad(z)
	return z

class SparseLabels:
	def __init__(labels):
		labels.indices = {}
		labels.values = []

	def shape(self):
		return (len(self.indices),len(self.values))

# labels: An `int32` `SparseTensor`.
# labels.indices[i, :] == [b, t] means `labels.values[i]` stores the id for (batch b, time t).
# labels.values[i]` must take on values in `[0, num_labels)`.
def sparse_labels(vec):
	labels = SparseLabels()
	b=0
	for lab in vec:
		t=0
		for c in lab:
			labels.indices[b, t] = len(labels.values)
			labels.values.append(char_to_class(c))
			# labels.values[i] = char_to_class(c)
			t += 1
		b += 1
	return labels



def progresshook(blocknum, blocksize, totalsize):
		readsofar = blocknum * blocksize
		if totalsize > 0:
				percent = readsofar * 1e2 / totalsize
				s = "\r%5.1f%% %*d / %d" % (
						percent, len(str(totalsize)), readsofar, totalsize)
				sys.stderr.write(s)
				if readsofar >= totalsize: # near the end
						sys.stderr.write("\n")
		else: # total size is unknown
				sys.stderr.write("read %d\n" % (readsofar,))

def maybe_download(file, work_directory=DATA_DIR):
	"""Download the data from Pannous's website, unless it's already here."""
	print("Looking for data %s in %s"%(file,work_directory))
	if not os.path.exists(work_directory):
		try:
			os.mkdir(work_directory)
		except:
			pass
	filepath = os.path.join(work_directory, re.sub('.*\/','',file))
	if not os.path.exists(filepath):
		if not file.startswith("http"): url_filename = SOURCE_URL + file
		else: url_filename=file
		print('Downloading from %s to %s' % (url_filename, filepath))
		filepath, _ = urllib.request.urlretrieve(url_filename, filepath,progresshook)
		statinfo = os.stat(filepath)
		print('Successfully downloaded', file, statinfo.st_size, 'bytes.')
		# os.system('ln -s '+work_directory)
	if os.path.exists(filepath):
		print('Extracting %s to %s' % ( filepath, work_directory))
		os.system('tar xf '+filepath+" -C "+work_directory)
		print('Data ready!')
	return filepath.replace(".tar","")

def spectro_batch(batch_size=10):
	return spectro_batch_generator(batch_size)

def speaker(filename):  # vom Dateinamen
	# if not "_" in file:
	#   return "Unknown"
	return filename.split("_")[1]

def get_speakers(path=pcm_path):
	maybe_download(Source.DIGIT_SPECTROS)
	maybe_download(Source.DIGIT_WAVES)
	files = os.listdir(path)
	def nobad(name):
		return "_" in name and not "." in name.split("_")[1]
	speakers=list(set(map(speaker,filter(nobad,files))))
	print(len(speakers)," speakers: ",speakers)
	return speakers

def load_wav_file(name):
	f = wave.open(name, "rb")
	# print("loading %s"%name)
	chunk = []
	data0 = f.readframes(CHUNK)
	while data0:  # f.getnframes()
		# data=numpy.fromstring(data0, dtype='float32')
		# data = numpy.fromstring(data0, dtype='uint16')
		data = numpy.fromstring(data0, dtype='uint8')
		data = (data + 128) / 255.  # 0-1 for Better convergence
		# chunks.append(data)
		chunk.extend(data)
		data0 = f.readframes(CHUNK)
	# finally trim:
	chunk = chunk[0:CHUNK * 2]  # should be enough for now -> cut
	chunk.extend(numpy.zeros(CHUNK * 2 - len(chunk)))  # fill with padding 0's
	# print("%s loaded"%name)
	return chunk


def spectro_batch_generator(batch_size=10,width=64,source_data=Source.DIGIT_SPECTROS,target=Target.digits):
	# maybe_download(Source.NUMBER_IMAGES , DATA_DIR)
	# maybe_download(Source.SPOKEN_WORDS, DATA_DIR)
	path=maybe_download(source_data, DATA_DIR)
	path=path.replace("_spectros","")# HACK! remove!
	height = width
	batch = []
	labels = []
	speakers=get_speakers(path)
	if target==Target.digits: num_classes=10
	if target==Target.first_letter: num_classes=32
	if target==Target.speech_commands: num_classes=len(speech_commands)
	files = os.listdir(path)
	# shuffle(files) # todo : split test_fraction batch here!
	# files=files[0:int(len(files)*(1-test_fraction))]
	print("Got %d source data files from %s"%(len(files),path))
	while True:
		# print("shuffling source data files")
		shuffle(files)
		for image_name in files:
			if not "_" in image_name: continue # bad !?!
			image = skimage.io.imread(path + "/" + image_name).astype(numpy.float32)
			# image.resize(width,height) # lets see ...
			data = image / 255.  # 0-1 for Better convergence
			# data = data.reshape([width * height])  # tensorflow matmul needs flattened matrices wtf
			batch.append(list(data))
			# classe=(ord(image_name[0]) - 48)  # -> 0=0 .. A:65-48 ... 74 for 'z'
			classe = (ord(image_name[0]) - 48) % 32# -> 0=0  17 for A, 10 for z ;)
			labels.append(dense_to_one_hot(classe,num_classes))
			if len(batch) >= batch_size:
				yield batch, labels
				batch = []  # Reset for next batch
				labels = []

def mfcc_batch_generator(batch_size=10, source=Source.DIGIT_WAVES, target=Target.digits):
	maybe_download(source, DATA_DIR)
	if target == Target.speaker: speakers = get_speakers()
	batch_features = []
	labels = []
	files = os.listdir(path)
	while True:
		print("loaded batch of %d files" % len(files))
		shuffle(files)
		for file in files:
			if not file.endswith(".wav"): continue
			wave, sr = librosa.load(path+file, mono=True)
			mfcc = librosa.feature.mfcc(wave, sr)
			if target==Target.speaker: label=one_hot_from_item(speaker(file), speakers)
			elif target==Target.digits:  label=dense_to_one_hot(int(file[0]),10)
			elif target==Target.first_letter:  label=dense_to_one_hot((ord(file[0]) - 48) % 32,32)
			elif target == Target.hotword: label = one_hot_word(file, pad_to=max_word_length)  #
			elif target == Target.word: label=string_to_int_word(file, pad_to=max_word_length)
				# label = file  # sparse_labels(file, pad_to=20)  # max_output_length
			else: raise Exception("todo : labels for Target!")
			labels.append(label)
			# print(np.array(mfcc).shape)
			mfcc=np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
			batch_features.append(np.array(mfcc))
			if len(batch_features) >= batch_size:
				# if target == Target.word:  labels = sparse_labels(labels)
				# labels=np.array(labels)
				# print(np.array(batch_features).shape)
				# yield np.array(batch_features), labels
				# print(np.array(labels).shape) # why (64,) instead of (64, 15, 32)? OK IFF dim_1==const (20)
				yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
				batch_features = []  # Reset for next batch
				labels = []


# If you set dynamic_pad=True when calling tf.train.batch the returned batch will be automatically padded with 0s. Handy! A lower-level option is to use tf.PaddingFIFOQueue.
# only apply to a subset of all images at one time
def wave_batch_generator(batch_size=10,source=Source.DIGIT_WAVES,target=Target.digits): #speaker
	maybe_download(source, DATA_DIR)
	if target == Target.speaker: speakers=get_speakers()
	batch_waves = []
	labels = []
	# input_width=CHUNK*6 # wow, big!!
	files = os.listdir(path)
	while True:
		shuffle(files)
		print("loaded batch of %d files" % len(files))
		for wav in files:
			if not wav.endswith(".wav"):continue
			if target==Target.digits: labels.append(dense_to_one_hot(int(wav[0])))
			elif target==Target.speaker: labels.append(one_hot_from_item(speaker(wav), speakers))
			elif target==Target.first_letter:  label=dense_to_one_hot((ord(wav[0]) - 48) % 32,32)
			else: raise Exception("todo : Target.word label!")
			chunk = load_wav_file(path+wav)
			batch_waves.append(chunk)
			# batch_waves.append(chunks[input_width])
			if len(batch_waves) >= batch_size:
				yield batch_waves, labels
				batch_waves = []  # Reset for next batch
				labels = []

class DataSet(object):

	def __init__(self, images, labels, fake_data=False, one_hot=False, load=False):
		"""Construct a DataSet. one_hot arg is used only if fake_data is true."""
		if fake_data:
			self._num_examples = 10000
			self.one_hot = one_hot
		else:
			num = len(images)
			assert num == len(labels), ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
			print("len(images) %d" % num)
			self._num_examples = num
		self.cache={}
		self._image_names = numpy.array(images)
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0
		self._images=[]
		if load: # Otherwise loaded on demand
			self._images=self.load(self._image_names)

	@property
	def images(self):
		return self._images

	@property
	def image_names(self):
		return self._image_names

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	# only apply to a subset of all images at one time
	def load(self,image_names):
		print("loading %d images"%len(image_names))
		return list(map(self.load_image,image_names)) # python3 map object WTF

	def load_image(self,image_name):
		if image_name in self.cache:
				return self.cache[image_name]
		else:
			image = skimage.io.imread(DATA_DIR+ image_name).astype(numpy.float32)
			# images = numpy.multiply(images, 1.0 / 255.0)
			self.cache[image_name]=image
			return image


	def next_batch(self, batch_size, fake_data=False):
		"""Return the next `batch_size` examples from this data set."""
		if fake_data:
			fake_image = [1] * width * height
			if self.one_hot:
				fake_label = [1] + [0] * 9
			else:
				fake_label = 0
			return [fake_image for _ in xrange(batch_size)], [
					fake_label for _ in xrange(batch_size)]
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = numpy.arange(self._num_examples)
			numpy.random.shuffle(perm)
			# self._images = self._images[perm]
			self._image_names = self._image_names[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self.load(self._image_names[start:end]), self._labels[start:end]


# multi-label
def dense_to_some_hot(labels_dense, num_classes=140):
	"""Convert class labels from int vectors to many-hot vectors!"""
	raise "TODO dense_to_some_hot"


def one_hot_to_item(hot, items):
	i=np.argmax(hot)
	item=items[i]
	return item

def one_hot_from_item(item, items):
	# items=set(items) # assure uniqueness
	x=[0]*len(items)# numpy.zeros(len(items))
	i=items.index(item)
	x[i]=1
	return x


def one_hot_word(word,pad_to=max_word_length):
	vec=[]
	for c in word:#.upper():
		x = [0] * num_characters
		x[(ord(c) - offset)%num_characters]=1
		vec.append(x)
	if pad_to:vec=pad(vec, pad_to, one_hot=True)
	return vec

def many_hot_to_word(word):
	s=""
	for c in word:
		x=np.argmax(c)
		s+=chr(x+offset)
		# s += chr(x + 48) # numbers
	return s


def dense_to_one_hot(batch, batch_size, num_labels):
	sparse_labels = tf.reshape(batch, [batch_size, 1])
	indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
	concatenated = tf.concat(axis=1, values=[indices, sparse_labels])
	concat = tf.concat(axis=0, values=[[batch_size], [num_labels]])
	output_shape = tf.reshape(concat, [2])
	sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
	return tf.reshape(sparse_to_dense, [batch_size, num_labels])


def dense_to_one_hot(batch, batch_size, num_labels):
	sparse_labels = tf.reshape(batch, [batch_size, 1])
	indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
	concatenated = tf.concat(axis=1, values=[indices, sparse_labels])
	concat = tf.concat(axis=0, values=[[batch_size], [num_labels]])
	output_shape = tf.reshape(concat, [2])
	sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
	return tf.reshape(sparse_to_dense, [batch_size, num_labels])

def dense_to_one_hot(labels_dense, num_classes=10):
	"""Convert class labels from scalars to one-hot vectors."""
	return numpy.eye(num_classes)[labels_dense]

def extract_labels(names_file,train, one_hot):
	labels=[]
	for line in open(names_file).readlines():
		image_file,image_label = line.split("\t")
		labels.append(image_label)
	if one_hot:
			return dense_to_one_hot(labels)
	return labels

def extract_images(names_file,train):
	image_files=[]
	for line in open(names_file).readlines():
		image_file,image_label = line.split("\t")
		image_files.append(image_file)
	return image_files


def read_data_sets(train_dir,source_data=Source.NUMBER_IMAGES, fake_data=False, one_hot=True):
	class DataSets(object):
		pass
	data_sets = DataSets()
	if fake_data:
		data_sets.train = DataSet([], [], fake_data=True, one_hot=one_hot)
		data_sets.validation = DataSet([], [], fake_data=True, one_hot=one_hot)
		data_sets.test = DataSet([], [], fake_data=True, one_hot=one_hot)
		return data_sets
	VALIDATION_SIZE = 2000
	local_file = maybe_download(source_data, train_dir)
	train_images = extract_images(TRAIN_INDEX,train=True)
	train_labels = extract_labels(TRAIN_INDEX,train=True, one_hot=one_hot)
	test_images = extract_images(TEST_INDEX,train=False)
	test_labels = extract_labels(TEST_INDEX,train=False, one_hot=one_hot)
	# train_images = train_images[:VALIDATION_SIZE]
	# train_labels = train_labels[:VALIDATION_SIZE:]
	# test_images = test_images[VALIDATION_SIZE:]
	# test_labels = test_labels[VALIDATION_SIZE:]
	data_sets.train = DataSet(train_images, train_labels , load=False)
	data_sets.test = DataSet(test_images, test_labels, load=True)
	# data_sets.validation = DataSet(validation_images, validation_labels, load=True)
	return data_sets

if __name__ == "__main__":
	print("downloading speech datasets")
	maybe_download( Source.DIGIT_SPECTROS)
	maybe_download( Source.DIGIT_WAVES)
	maybe_download( Source.NUMBER_IMAGES)
	maybe_download( Source.NUMBER_WAVES)
