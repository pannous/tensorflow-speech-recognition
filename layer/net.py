from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector  # for 3d PCA/ t-SNE

from .tensorboard_util import *

start = int(time.time())

# clear_tensorboard()
set_tensorboard_run(auto_increment=True)
run_tensorboard(restart=False)

# gpu = True
gpu = False
debug = False  # True # summary.histogram  : 'module' object has no attribute 'histogram' WTF
debug = True  # histogram_summary ...
visualize_cluster = False  # NOT YET: 'ProjectorConfig' object has no attribute 'embeddings'

slim = tf.contrib.slim
weight_divider = 10.
default_learning_rate = 0.001  # mostly overwritten, so ignore it
decay_steps = 100000
decay_size = 0.1
save_step = 10000  # if you don't want to save snapshots, set to -1
checkpoint_dir = "checkpoints"

if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)


def nop(): return 0


def closest_unitary(A):
	""" Calculate the unitary matrix U that is closest with respect to the operator norm distance to the general matrix A. """
	import scipy
	V, __, Wh = scipy.linalg.svd(A)
	return np.matrix(V.dot(Wh))


_cpu = '/cpu:0'


class net():
	def __init__(self, model, data=0, input_width=0, output_width=0, input_shape=0, name=0,
	             learning_rate=default_learning_rate):
		device = '/GPU:0' if gpu else '/cpu:0'
		device = None  # auto
		print("Using device ", device)
		with tf.device(device):
			# if True:
			self.session = sess = session = tf.Session()
			# self.session=sess=session=tf.Session(config=tf.ConfigProto(log_device_placement=True))
			self.model = model
			self.input_shape = input_shape or [input_width, input_width]
			self.data = data  # assigned to self.x=net.input via train
			if not input_width:
				input_width = self.get_data_shape()
			self.input_width = input_width
			self.last_width = self.input_width
			self.output_width = output_width
			self.num_classes = output_width
			# self.batch_size=batch_size
			self.layers = []
			self.learning_rate = learning_rate
			if not name: name = model.__name__
			self.name = str(name)
			if name and os.path.exists(name + ".model"):
				return self.load_model(name + ".model")
			self.generate_model(model)

	def get_data_shape(self):
		if self.input_shape:
			return self.input_shape[0], self.input_shape[1]
		try:
			return self.data.shape[0], self.data.shape[-1]
		except:
			raise Exception("Data does not have shape")

	def generate_model(self, model, name=''):
		if not model: return self
		with tf.name_scope('state'):
			self.keep_prob = tf.placeholder(tf.float32)  # 1 for testing! else 1 - dropout
			self.train_phase = tf.placeholder(tf.bool, name='train_phase')
			with tf.device(_cpu): self.global_step = tf.Variable(
				0)  # dont set, feed or increment global_step, tensorflow will do it automatically
		with tf.name_scope('data'):
			if len(self.input_shape) == 1:
				self.input_width = self.input_shape[0]
			elif self.input_shape:
				self.x = x = self.input = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1]])
				# todo [None, self.input_shape]
				self.last_layer = x
				self.last_shape = x
			elif self.input_width:
				self.x = x = self.target = tf.placeholder(tf.float32, [None, self.input_width])
				self.last_layer = x
			else:
				raise Exception("need input_shape or input_width by now")
			self.y = y = self.target = tf.placeholder(tf.float32, [None, self.output_width])
		with tf.name_scope('model'):
			model(self)
		if (self.last_width != self.output_width):
			self.classifier()  # 10 classes auto

	def dropout(self, keep_rate=0.6):
		self.add(tf.nn.dropout(self.last_layer, keep_rate))

	def fully_connected(self, hidden=1024, depth=1, activation=tf.nn.tanh, dropout=False, parent=-1, norm=None):  # ):
		return self.dense()

	# Fully connected 'pyramid' layer, allows very high learning_rate >0.1 (but don't abuse)
	def denseNet(self, hidden=20, depth=3, act=tf.nn.tanh, dropout=True, norm=None):  #
		if (hidden > 100): print("WARNING: denseNet uses quadratic mem for " + str(hidden))
		if (depth < 3): print(
			"WARNING: did you mean to use Fully connected layer 'dense'? Expecting depth>3 vs " + str(depth))
		inputs = self.last_layer
		inputs_width = self.last_width
		width = hidden
		while depth > 0:
			with tf.name_scope('DenNet_{:d}'.format(width)) as scope:
				print("dense width ", inputs_width, "x", width)
				nr = len(self.layers)
				weights = tf.Variable(tf.random_uniform([inputs_width, width], minval=-1. / width, maxval=1. / width),
				                      name="weights")
				bias = tf.Variable(tf.random_uniform([width], minval=-1. / width, maxval=1. / width),
				                   name="bias")  # auto nr + context
				dense1 = tf.matmul(inputs, weights, name='dense_' + str(nr)) + bias
				tf.summary.histogram('dense_' + str(nr), dense1)
				tf.summary.histogram('dense_' + str(nr) + '/sparsity', tf.nn.zero_fraction(dense1))
				tf.summary.histogram('weights_' + str(nr), weights)
				tf.summary.histogram('weights_' + str(nr) + '/sparsity', tf.nn.zero_fraction(weights))
				tf.summary.histogram('bias_' + str(nr), bias)

				if act: dense1 = act(dense1)
				if norm: dense1 = self.norm(dense1, lsize=1)  # SHAPE!
				if dropout: dense1 = tf.nn.dropout(dense1, self.keep_prob)
				self.add(dense1)
				self.last_width = width
				inputs = tf.concat(1, [inputs, dense1])
				inputs_width += width
				depth = depth - 1
		self.last_width = width

	def add(self, layer):
		self.layers.append(layer)
		self.last_layer = layer
		self.last_shape = layer.get_shape()

	def reshape(self, shape):
		self.last_layer = tf.reshape(self.last_layer, shape)
		self.last_shape = shape
		self.last_width = shape[-1]

	def batchnorm(self):
		from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
		with tf.name_scope('batchnorm') as scope:
			input = self.last_layer
			# mean, var = tf.nn.moments(input, axes=[0, 1, 2])
			# self.batch_norm = tf.nn.batch_normalization(input, mean, var, offset=1, scale=1, variance_epsilon=1e-6)
			# self.last_layer=self.batch_norm
			train_op = batch_norm(input, is_training=True, center=False, updates_collections=None, scope=scope)
			test_op = batch_norm(input, is_training=False, updates_collections=None, center=False, scope=scope,
			                     reuse=True)
			self.add(tf.cond(self.train_phase, lambda: train_op, lambda: test_op))

	def addLayer(self, nChannels, nOutChannels, do_dropout):
		ident = self.last_layer
		self.batchnorm()
		# self.add(tf.nn.relu(ident)) # nChannels ?
		self.conv([3, 3, nChannels, nOutChannels], pool=False, dropout=do_dropout, norm=tf.nn.relu)  # None
		concat = tf.concat(3, [ident, self.last_layer])
		print("concat ", concat.get_shape())
		self.add(concat)

	def addTransition(self, nChannels, nOutChannels, do_dropout):
		self.batchnorm()
		self.add(tf.nn.relu(self.last_layer))
		self.conv([1, 1, nChannels, nOutChannels], pool=True, dropout=do_dropout, norm=None)  # pool (2, 2)

	# self.add(tf.nn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))

	# Densely Connected Convolutional Networks https://arxiv.org/abs/1608.06993
	def buildDenseConv(self, N_blocks=3, magic_factor=1):
		depth = 3 * N_blocks + 4
		if (depth - 4) % 3:  raise Exception("Depth must be 3N + 4! (4,7,10,...) ")  # # layers in each denseblock
		N = (depth - 4) / 3
		do_dropout = True  # None  nil to disable dropout, non - zero number to enable dropout and set drop rate
		# dropRate = self.keep_prob # nil to disable dropout, non - zero number to enable dropout and set drop rate
		# # channels before entering the first denseblock ??
		# set it to be comparable with growth rate ??
		nChannels = 64
		# nChannels = 16
		growthRate = 12
		self.conv([3, 3, 1, nChannels])
		# self.add(tf.nn.SpatialConvolution(3, nChannels, 3, 3, 1, 1, 1, 1))

		for i in range(N):
			self.addLayer(nChannels, growthRate, do_dropout)
			nChannels = nChannels + growthRate
		self.addTransition(nChannels, nChannels, do_dropout)

		for i in range(N):
			self.addLayer(nChannels, growthRate, do_dropout)
			nChannels = nChannels + growthRate
		self.addTransition(nChannels, nChannels, do_dropout)

		for i in range(N):
			self.addLayer(nChannels, growthRate, do_dropout)
			nChannels = nChannels + growthRate

		self.batchnorm()
		self.add(tf.nn.relu(self.last_layer))
		# self.add(tf.nn.max_pool(self.last_layer, ksize=[1, 8, 8, 1], strides=[1, 2, 2, 1], padding='SAME'))
		# self.add(tf.nn.max_pool(self.last_layer, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='SAME'))
		# self.add(tf.nn.max_pool(self.last_layer, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME'))
		self.add(tf.nn.max_pool(self.last_layer, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME'))
		# self.add(tf.nn.SpatialAveragePooling(8, 8)).add(nn.Reshape(nChannels))
		if magic_factor == 16:
			self.reshape([-1, nChannels * 16])  # ready for classification
		else:
			self.reshape([-1, nChannels * 4])  # ready for classification

	# Fully connected layer
	def dense(self, hidden=1024, depth=1, activation=tf.nn.tanh, dropout=False, parent=-1, norm=None):  #
		if parent == -1: parent = self.last_layer
		shape = self.last_layer.get_shape()
		if shape and len(shape) > 2:
			if len(shape) == 3:
				self.last_width = int(shape[1] * shape[2])
			else:
				self.last_width = int(shape[1] * shape[2] * shape[3])
			print("reshapeing ", shape, "to", self.last_width)
			parent = tf.reshape(parent, [-1, self.last_width])

		width = hidden
		while depth > 0:
			with tf.name_scope('Dense_{:d}'.format(hidden)) as scope:
				print("Dense ", self.last_width, width)
				nr = len(self.layers)
				if self.last_width == width:
					U = closest_unitary(
						np.random.rand(self.last_width, width) / (self.last_width + width)) / weight_divider
					weights = tf.Variable(U, name="weights_dense_" + str(nr), dtype=tf.float32)
				else:
					weights = tf.Variable(
						tf.random_uniform([self.last_width, width], minval=-1. / width, maxval=1. / width),
						name="weights_dense")
				bias = tf.Variable(tf.random_uniform([width], minval=-1. / width, maxval=1. / width), name="bias_dense")
				dense1 = tf.matmul(parent, weights, name='dense_' + str(nr)) + bias
				tf.summary.histogram('dense_' + str(nr), dense1)
				tf.summary.histogram('weights_' + str(nr), weights)
				tf.summary.histogram('bias_' + str(nr), bias)
				tf.summary.histogram('dense_' + str(nr) + '/sparsity', tf.nn.zero_fraction(dense1))
				tf.summary.histogram('weights_' + str(nr) + '/sparsity', tf.nn.zero_fraction(weights))
				if activation: dense1 = activation(dense1)
				if norm: dense1 = self.norm(dense1, lsize=1)
				if dropout: dense1 = tf.nn.dropout(dense1, self.keep_prob)
				self.layers.append(dense1)
				self.last_layer = parent = dense1
				self.last_width = width
				depth = depth - 1
				self.last_shape = [-1, width]  # dense

	def conv2(self, shape, act=tf.nn.relu, pool=True, dropout=False, norm=True, name=None):
		with tf.name_scope('conv'):
			print("input  shape ", self.last_shape)
			print("conv   shape ", shape)
			# padding='VALID'
			conv = slim.conv2d(self.last_layer, shape[-1], [shape[1], shape[2]], 3, padding='SAME', scope=name)
			# if pool: conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			# if(pool): conv = slim.max_pool2d(conv, [2, 2], 1, scope='pool1')
			# if(pool): conv = slim.max_pool2d(conv, [3, 3], 2, scope='pool1')
			self.add(conv)

	# Convolution Layer
	def conv(self, shape, act=tf.nn.relu, pool=True, dropout=False, norm=True,
	         name=None):  # True why dropout bad in tensorflow??
		with tf.name_scope('conv'):
			print("input  shape ", self.last_shape)
			print("conv   shape ", shape)
			width = shape[-1]
			filters = tf.Variable(tf.random_normal(shape), name="filters")
			# filters = tf.Variable(tf.random_uniform(shape, minval=-1. / width, maxval=1. / width), name="filters")
			_bias = tf.Variable(tf.random_normal([shape[-1]]), name="bias")

			# # conv1 = conv2d('conv', _X, _weights, _bias)
			conv1 = tf.nn.bias_add(tf.nn.conv2d(self.last_layer, filter=filters, strides=[1, 1, 1, 1], padding='SAME'),
			                       _bias)
			if debug: tf.summary.histogram('conv_' + str(len(self.layers)), conv1)
			if act: conv1 = act(conv1)
			if pool: conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			if norm: conv1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
			if debug: tf.summary.histogram('norm_' + str(len(self.layers)), conv1)
			if dropout: conv1 = tf.nn.dropout(conv1, self.keep_prob)
			print("output shape ", conv1.get_shape())
			self.add(conv1)

	def rnn(self):
		# data = tf.placeholder(tf.float32, [None, width, height])  # Number of examples, number of input, dimension of each input
		# target = tf.placeholder(tf.float32, [None, classes])
		# num_hidden = 24
		num_hidden = 42
		cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
		val, _ = tf.nn.dynamic_rnn(cell, self.last_layer, dtype=tf.float32)
		val = tf.nn.dropout(val, 0.8)
		val = tf.transpose(val, [1, 0, 2])
		self.last = tf.gather(val, int(val.get_shape()[0]) - 1)

	# weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
	# bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
	# mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
	# error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

	def classifier(self, classes=0):  # Define loss and optimizer
		if not classes: classes = self.num_classes
		with tf.name_scope('prediction'):  # prediction
			if self.last_width != classes:
				# print("Automatically adding dense prediction")
				self.dense(hidden=classes, activation=False, dropout=False)
			# cross_entropy = -tf.reduce_sum(y_*y)
		with tf.name_scope('classifier'):
			y_ = self.target
			manual = False  # True
			if classes > 100:
				print("using sampled_softmax_loss")
				y = prediction = self.last_layer
				self.cost = tf.reduce_mean(tf.nn.sampled_softmax_loss(y, y_))  # for big vocab
			elif manual:
				# prediction = y =self.last_layer=tf.nn.softmax(self.last_layer)
				# self.cost = cross_entropy = -tf.reduce_sum(y_ * tf.log(y+ 1e-10)) # against NaN!
				prediction = y = tf.nn.log_softmax(self.last_layer)
				self.cost = cross_entropy = -tf.reduce_sum(y_ * y)
			else:
				y = prediction = self.last_layer
				self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))  # prediction, target

			# if not gpu:
			with tf.device(_cpu):
				tf.summary.scalar('cost', self.cost)
			# self.cost = tf.Print(self.cost , [self.cost ], "debug cost : ")
			# learning_scheme=self.learning_rate
			learning_scheme = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_size,
			                                             staircase=True)
			with tf.device(_cpu):
				tf.summary.scalar('learning_rate', learning_scheme)
			self.optimizer = tf.train.AdamOptimizer(learning_scheme).minimize(self.cost)
			# self.optimizer = NeuralOptimizer(data=None, learning_rate=0.01, shared_loss=self.cost).minimize(self.cost) No good

			# Evaluate model
			correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.target, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			if not gpu: tf.summary.scalar('accuracy', self.accuracy)
		# Launch the graph

	def next_batch(self, batch_size, session, test=False):
		try:
			if test:
				test_images = self.data.test.images[:batch_size]
				test_labels = self.data.test.labels[:batch_size]
				return test_images, test_labels
			return self.data.train.next_batch(batch_size)
		except:
			try:
				return next(self.data)
			except:
				return next(self.data.train)

	def train(self, data=0, steps=-1, dropout=None, display_step=10, test_step=200, batch_size=10,
	          do_resume=False):  # epochs=-1,
		if data: self.data = data
		steps = 9999999 if steps == -1 else steps
		session = self.session
		# with tf.device(_cpu):

		# import tensorflow.contrib.layers as layers
		# t = tf.verify_tensor_all_finite(t, msg)
		tf.add_check_numerics_ops()
		try:
			self.summaries = tf.summary.merge_all()
		except:
			self.summaries = tf.merge_all_summaries()
		try:
			self.summary_writer = tf.summary.FileWriter(current_logdir(), session.graph)  #
		except:
			self.summary_writer = tf.train.SummaryWriter(current_logdir(), session.graph)  #
		if not dropout: dropout = 1.  # keep all
		x = self.x
		y = self.y
		keep_prob = self.keep_prob
		try:
			saver = tf.train.Saver(tf.global_variables())
		except:
			saver = tf.train.Saver(tf.all_variables())
		snapshot = self.name + str(get_last_tensorboard_run_nr())
		checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
		if do_resume and checkpoint:
			print("LOADING " + checkpoint + " !!!")
			saver.restore(session, checkpoint)
		try:
			session.run([tf.global_variables_initializer()])
		except:
			session.run([tf.initialize_all_variables()])
		step = 0  # show first
		while step < steps:
			batch_xs, batch_ys = self.next_batch(batch_size, session)
			# print("step %d \r" % step)# end=' ')

			# tf.train.shuffle_batch_join(example_list, batch_size, capacity=min_queue_size + batch_size * 16, min_queue_size)
			# Fit training using batch data
			feed_dict = {x: batch_xs, y: batch_ys, keep_prob: dropout, self.train_phase: True}
			loss, _ = session.run([self.cost, self.optimizer], feed_dict=feed_dict)
			if step % display_step == 0:
				seconds = int(time.time()) - start
				# Calculate batch accuracy, loss
				feed = {x: batch_xs, y: batch_ys, keep_prob: 1., self.train_phase: False}
				acc, summary = session.run([self.accuracy, self.summaries], feed_dict=feed)
				# self.summary_writer.add_summary(summary, step) # only test summaries for smoother curve
				print("\rStep {:d} Loss= {:.6f} Accuracy= {:.3f} Time= {:d}s".format(step, loss, acc, seconds), end=' ')
				if str(loss) == "nan": return print("\nLoss gradiant explosion, exiting!!!")  # restore!
			if step % test_step == 0: self.test(step)
			if step % save_step == 0 and step > 0:
				print("SAVING snapshot %s" % snapshot)
				saver.save(session, checkpoint_dir + snapshot + ".ckpt", self.global_step)

			step += 1
		print("\nOptimization Finished!")
		self.test(step, number=10000)  # final test

	def test(self, step, number=400):  # 256 self.batch_size
		session = sess = self.session
		config = projector.ProjectorConfig()
		if visualize_cluster:
			embedding = config.embeddings.add()  # You can add multiple embeddings. Here just one.
			embedding.tensor_name = self.last_layer.name  # last_dense
			# embedding.tensor_path
			# embedding.tensor_shape
			embedding.sprite.image_path = PATH_TO_SPRITE_IMAGE
			# help(embedding.sprite)
			embedding.sprite.single_image_dim.extend([width, hight])  # if mnist   thumbnail
			# embedding.single_image_dim.extend([28, 28]) # if mnist   thumbnail
			# Link this tensor to its metadata file (e.g. labels).
			embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
			# Saves a configuration file that TensorBoard will read during startup.
			projector.visualize_embeddings(self.summary_writer, config)

		run_metadata = tf.RunMetadata()
		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		# Calculate accuracy for 256 mnist test images

		test_images, test_labels = self.next_batch(number, session, test=True)

		feed_dict = {self.x: test_images, self.y: test_labels, self.keep_prob: 1., self.train_phase: False}
		# accuracy,summary= self.session.run([self.accuracy, self.summaries], feed_dict=feed_dict)
		accuracy, summary = session.run([self.accuracy, self.summaries], feed_dict, run_options, run_metadata)
		print('\t' * 3 + "Test Accuracy: ", accuracy)
		self.summary_writer.add_run_metadata(run_metadata, 'step #%03d' % step)
		self.summary_writer.add_summary(summary, global_step=step)

	# def inputs(self,data):
	# 	self.inputs, self.labels = load_data()#...)
