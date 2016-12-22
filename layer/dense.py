import tensorflow as tf
import layer

class net(layer.net):

	# Fully connected 'pyramid' layer, allows very high learning_rate!
	def fullDenseNet(self, hidden=20, depth=3, act=tf.nn.tanh, dropout=True): #
		if(hidden>100):print("WARNING: denseNet uses quadratic mem for "+str(hidden))
		if(depth<3):print("WARNING: did you mean to use Fully connected layer 'dense'? Expecting depth>3 vs "+str(depth))
		inputs=self.last_layer
		inputs_width=self.last_width
		width = hidden
		while depth>0:
			with tf.name_scope('DenNet_{:d}'.format(width)) as scope:
				print("dense width ",inputs_width,"x",width)
				nr = len(self.layers)
				weights = tf.Variable(tf.random_uniform([inputs_width, width],minval=-1./width,maxval=1./width), name="weights_dense_" + str(nr))
				bias = tf.Variable(tf.random_uniform([width],minval=-1./width,maxval=1./width), name="bias_dense_" + str(nr))
				dense1 = tf.matmul(inputs, weights, name='dense_'+str(nr))+ bias
				tf.histogram_summary('dense_'+str(nr),dense1)
				tf.histogram_summary('weights_'+str(nr),weights)
				tf.histogram_summary('bias_'+str(nr),bias)
				if act: dense1 = act(dense1)
				# if norm: dense1 = self.norm(dense1,lsize=1) # SHAPE!
				if dropout: dense1 = tf.nn.dropout(dense1, self.keep_prob)
				self.add(dense1)
				self.last_width = width
				inputs=tf.concat(1,[inputs,dense1])
				inputs_width+=width
				depth=depth-1
		self.last_width = width


	# Fully connected layer
	def dense(self, hidden=1024, depth=1, act=tf.nn.tanh, dropout=True, parent=-1):  #
		if parent == -1: parent = self.last_layer
		shape = self.last_layer.get_shape()
		if shape and len(shape) > 2:
			self.last_width = int(shape[1] * shape[2] * shape[3])
			print("reshapeing ", shape, "to", self.last_width)
			parent = tf.reshape(parent, [-1, self.last_width])

		width = hidden
		while depth > 0:
			with tf.name_scope('Dense_{:d}'.format(hidden)) as scope:
				print("Dense ", self.last_width, width)
				nr = len(self.layers)
				weights = tf.Variable(tf.random_uniform([self.last_width, width], minval=-1. / width, maxval=1. / width),
				                      name="weights_dense_" + str(nr))
				bias = tf.Variable(tf.random_uniform([width], minval=-1. / width, maxval=1. / width),
				                   name="bias_dense_" + str(nr))
				dense1 = tf.matmul(parent, weights, name='dense_' + str(nr)) + bias
				# tf.histogram_summary('dense_'+str(nr),dense1)
				# tf.histogram_summary('weights_'+str(nr),weights)
				# tf.histogram_summary('bias_'+str(nr),bias)
				if act: dense1 = act(dense1)
				# if norm: dense1 = self.norm(dense1,lsize=1) # SHAPE!
				if dropout: dense1 = tf.nn.dropout(dense1, self.keep_prob)
				self.layers.append(dense1)
				self.last_layer = parent = dense1
				self.last_width = width
				depth = depth - 1
				self.last_shape = [-1, width]  # dense
