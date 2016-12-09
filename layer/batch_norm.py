import layer

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

class net(layer.net):
	def batchnorm(self):
		isTraining=self.train_phase
		inputT=self.last_layer
		scope=None
		with tf.name_scope('batchnorm') as scope:
			return tf.cond(isTraining,
		               lambda: batch_norm(inputT, is_training=True,
		                                  center=False, updates_collections=None, scope=scope),
		               lambda: batch_norm(inputT, is_training=False,
		                                  updates_collections=None, center=False, scope=scope, reuse=True))

# 		scope_bn=None
# 		with tf.name_scope('batchnorm') as scope_bn:
# 			# beta=tf.Variable(0.5,name='beta')
# 			beta =tf.get_variable('beta',[1],dtype=tf.float32)
# 			x=self.last_layer
# 			# reuse=None#  ValueError: reuse=True cannot be used without a name_or_scope
# 			reuse =True
# 			bn_train = batch_norm(x, decay=0.999, center=True, scale=True, updates_collections=None, is_training=True, reuse=reuse,  trainable=True, scope=scope_bn)
# # ValueError: Variable model/conv/batchnorm//beta does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?
# 			bn_inference = \
# 				batch_norm(x,
# 			           decay=0.999,
# 			           center=True,
# 			           scale=True,
# 			           updates_collections=None,
# 			           is_training=False,
# 			           reuse=reuse,
# 			           trainable=True,
# 			           scope=scope_bn)
# 			z = tf.cond(self.train_phase, lambda: bn_train, lambda: bn_inference)
# 			return z



