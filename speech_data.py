"""Utilities for downloading and providing data from openslr.org, libriSpeech, Pannous, Gutenberg, WMT, tokenizing, vocabularies."""
# TODO! see https://github.com/pannous/caffe-speech-recognition for some data sources

import gzip
import os
import re
import skimage.io # scikit-image
import numpy
import numpy as np
import wave
# import extensions as xx
from random import shuffle
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin


# SOURCE_URL = 'https://www.dropbox.com/s/eb5zqskvnuj0r78/spoken_words.tar?dl=0'
# http://pannous.net/files/spoken_numbers_pcm.tar
SOURCE_URL = 'http://pannous.net/files/' #spoken_numbers.tar'
NUMBER_IMAGES = 'spoken_numbers.tar'
NUMBER_WAVES = 'spoken_numbers_wav.tar'
DIGIT_WAVES = 'spoken_numbers_pcm.tar'
DIGIT_SPECTROS = 'spoken_numbers_spectros_64x64.tar'
TEST_INDEX='test_index.txt'
TRAIN_INDEX='train_index.txt'
DATA_DIR='data'
# TRAIN_INDEX='train_words_index.txt'
# TEST_INDEX='test_words_index.txt'
# width=256
# height=256
width=512 # todo: sliding window!
height=512

def maybe_download(filename, work_directory):
  """Download the data from Pannous's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    print('Downloading %s from %s to %s' % ( filename, SOURCE_URL, filepath))
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    # os.system('ln -s '+work_directory)
  # if os.path.exists(filepath):
    print('Extracting %s to %s' % ( filepath, work_directory))
    os.system('tar xf '+filepath)
  return filepath


def spectro_batch(batch_size=10):
  return spectro_batch_generator(batch_size)


def spectro_batch_generator(batch_size,width=64):
  height=width
  batch = []
  labels = []
  # path = "data/spoken_numbers/"
  path = "data/spoken_numbers_64x64/"
  files=os.listdir(path)
  while True:
    shuffle(files)
    for image_name in files:
      image = skimage.io.imread(path+image_name).astype(numpy.float32)
      # image.resize(width,height) # lets see ...
      data = image/255. # 0-1 for Better convergence
      data = data.reshape([width*height]) # tensorflow matmul needs flattened matrices wtf
      batch.append(list(data))
      labels.append(dense_to_one_hot(int(image_name[0])))
      if len(batch) >= batch_size:
        yield batch, labels
        batch = []  # Reset for next batch
        labels = []

from enum import Enum
class Target(Enum):  # labels
  digits=1
  speaker=2
  words_per_minute=3


pcm_path = "data/spoken_numbers_pcm/" # 8 bit
wav_path = "data/spoken_numbers_wav/" # 16 bit s16le
path = pcm_path
CHUNK = 4096

def speaker(wav):  # vom Dateinamen
  return re.sub(r'_.*', '', wav[2:])

def get_speakers():
  files = os.listdir(pcm_path)
  return list(set(map(speaker,files)))

def load_wav_file(name):
  f = wave.open(name, "rb")
  chunk = []
  data0 = f.readframes(CHUNK)
  while data0 != '':  # f.getnframes()
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
  return chunk

# If you set dynamic_pad=True when calling tf.train.batch the returned batch will be automatically padded with 0s. Handy! A lower-level option is to use tf.PaddingFIFOQueue.
# only apply to a subset of all images at one time
def wave_batch_generator(batch_size=10,target=Target.speaker):
  maybe_download(DIGIT_WAVES, DATA_DIR)
  batch_waves = []
  labels = []
  # input_width=CHUNK*6 # wow, big!!
  speakers=get_speakers()
  files = os.listdir(path)
  while True:
    shuffle(files)
    for wav in files:
      if not wav.endswith(".wav"):continue
      if target==Target.digits: labels.append(dense_to_one_hot(int(wav[0])))
      if target==Target.speaker: labels.append(one_hot_from_item(speaker(wav), speakers))
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
      image = skimage.io.imread(image_name).astype(numpy.float32)
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
  pass


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

def dense_to_one_hot(batch, batch_size, num_labels):
  sparse_labels = tf.reshape(batch, [batch_size, 1])
  indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
  concatenated = tf.concat(1, [indices, sparse_labels])
  concat = tf.concat(0, [[batch_size], [num_labels]])
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


def read_data_sets(train_dir, fake_data=False, one_hot=True):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True, one_hot=one_hot)
    data_sets.validation = DataSet([], [], fake_data=True, one_hot=one_hot)
    data_sets.test = DataSet([], [], fake_data=True, one_hot=one_hot)
    return data_sets

  VALIDATION_SIZE = 2000

  local_file = maybe_download(NUMBER_IMAGES, train_dir)
  train_images = extract_images(TRAIN_INDEX,train=True)
  train_labels = extract_labels(TRAIN_INDEX,train=True, one_hot=one_hot)
  test_images = extract_images(TEST_INDEX,train=False)
  test_labels = extract_labels(TEST_INDEX,train=False, one_hot=one_hot)

  # validation_images = train_images[:VALIDATION_SIZE]
  # validation_labels = train_labels[:VALIDATION_SIZE]
  # train_images = train_images[VALIDATION_SIZE:]
  # train_labels = train_labels[VALIDATION_SIZE:]
  # validation_images = test_images[:VALIDATION_SIZE]
  # validation_labels = test_labels[:VALIDATION_SIZE]
  test_images = test_images[VALIDATION_SIZE:]
  test_labels = test_labels[VALIDATION_SIZE:]

  data_sets.train = DataSet(train_images, train_labels , load=False)
  # data_sets.validation = DataSet(validation_images, validation_labels, load=True)
  data_sets.test = DataSet(test_images, test_labels, load=True)

  return data_sets

if __name__ == "__main__":
  print("downloading speech datasets")
  maybe_download( SOURCE_URL + DIGIT_SPECTROS)
  maybe_download( SOURCE_URL + DIGIT_WAVES)
  maybe_download( SOURCE_URL + NUMBER_IMAGES)
  maybe_download( SOURCE_URL + NUMBER_WAVES)
