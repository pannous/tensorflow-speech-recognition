import tensorflow as tf
import layer

class net(layer.net):

	def conv_slim(w, b,pool=False):
		net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',scope='conv1')
		if(pool): net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
		self.last_layer = net

	# Convolution Layer
	def conv(self, shape, act=tf.nn.relu, pool=True, dropout=True, norm=True, name=None):
		with tf.name_scope('conv'):
			self.batchnorm()
			print("conv ", shape)
			filter_weights = tf.Variable(tf.random_normal(shape))  # WTF?
			_bias = tf.Variable(tf.random_normal([shape[-1]]))
			# conv1 = conv2d('conv', _X, _weights, _bias)
			conv1 = tf.nn.bias_add(tf.nn.conv2d(self.last_layer, filter=filter_weights, strides=[1, 1, 1, 1], padding='SAME'),
			                       _bias)
			if act: conv1 = act(conv1)
			# Max Pooling (down-sampling)
			if pool: conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			if norm: conv1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
			if dropout: conv1 = tf.nn.dropout(conv1, self.dropout_rate)
			self.last_layer = conv1
		# self.last_shape = conv1.get_shape()
