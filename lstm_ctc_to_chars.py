#!/usr/bin/env python
#!/usr/local/bin/python
from __future__ import print_function
import numpy as np
import tensorflow as tf
import speech_data
from speech_data import Source, Target
from tensorflow.python.ops import ctc_ops as ctc
# import layer
# from layer import net
import time

start = int(time.time())
display_step = 1
test_step = 10
save_step = 100
learning_rate = 0.0001
# 0.0001 Step 300 Loss= 1.976625 Accuracy= 0.250 Time= 303s
# Step 24261 Loss= 0.011786 Accuracy= 1.000 Time= 33762s takes time but works

training_iters = 300000  # steps
batch_size = 64

width = features = 20  # mfcc input features
height = max_input_length = 80  # (max) length of input utterance (mfcc slices)
classes = num_characters = 32
max_word_length = 20  # max length of output (characters per word)
# classes=10 # digits

keep_prob = dropout = 0.7

# batch = speech_data.mfcc_batch_generator(batch_size, target=Target.word)
batch = speech_data.mfcc_batch_generator(batch_size, source=Source.WORD_WAVES, target=Target.hotword)
X, Y = next(batch)
print("lable shape", np.array(Y).shape)

# inputs=tf.placeholder(tf.float32, shape=(batch_size,max_length,features))
x = inputX = inputs = tf.placeholder(tf.float32, shape=(batch_size, features, max_input_length))
# inputs = tf.transpose(inputs, [0, 2, 1]) #  inputs must be a `Tensor` of shape: `[batch_size, max_time, ...]`
inputs = tf.transpose(inputs, [2, 0, 1])  # [max_time, batch_size, features] to split:
# Split data because rnn cell needs a list of inputs for the RNN inner loop
inputs = tf.split(axis=0, num_or_size_splits=max_input_length, value=inputs)  # n_steps * (batch_size, features)

num_hidden = 100  # features
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
# cell = tf.nn.rnn_cell.EmbeddingWrapper(num_hidden, state_is_tuple=True)

# in many cases it may be more efficient to not use this wrapper,
#   but instead concatenate the whole sequence of your outputs in time,
#   do the projection on this batch-concatenated sequence, then split it
#   if needed or directly feed into a softmax.
# cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell,)


cell = tf.nn.rnn_cell.MultiRNNCell(num_hidden, state_is_tuple=True)
# rnn=tf.nn.rnn(cell,inputs)
# rnn=tf.nn.dynamic_rnn(cell,inputs)
# manual:

state = cell.zero_state(batch_size, dtype=tf.float32)
if "manual" == 0:
	outputs = []
	for input_ in inputs:
		input_ = tf.reshape(input_, [batch_size, features])
		output, state = cell(input_, state)
		outputs.append(output)
	y_ = output
else:
	# inputs = tf.reshape(inputs, [-1, features])
	inputs = [tf.reshape(input_, [batch_size, features]) for input_ in inputs]
	outputs, states = tf.nn.rnn(cell, inputs, initial_state=state)
# only last output as target for now
# y_=outputs[-1]

# optimize
target_shape = (batch_size, max_word_length, classes)
y = target = tf.placeholder(tf.float32, shape=target_shape)  # -> seq2seq!

# dense
logits = []
costs = []
i = 0
accuracy = 0
# for output in outputs:
for i in range(0, max_word_length):
	output = outputs[-i - 1]
	uniform = tf.random_uniform([num_hidden, classes], minval=-1. / width, maxval=1. / width)
	weights = tf.Variable(uniform, name="weights_%d" % i)
	uniform_bias = tf.random_uniform([classes], minval=-1. / width, maxval=1. / width)
	bias = tf.Variable(uniform_bias, name="bias_dense_%d" % i)
	y_ = outputY = tf.matmul(output, weights, name="dense_%d" % i) + bias

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y[:, i, :]), name="cost")  # prediction, target
	costs.append(cost)
	logits.append(y_)

	correct_pred = tf.equal(tf.argmax(outputY, 1), tf.argmax(y[:, i], 1))
	accuraci = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	accuracy += accuraci

# costs=tf.reduce_sum(costs)*10
# y_ = outputY = tf.pack(logits)

# targetIxs = tf.placeholder(tf.int64, shape=(batch_size, None),name="indices")
# targetVals = tf.placeholder(tf.int32,name="values")
# targetShape = tf.placeholder(tf.int64,name="targetShape")
# targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
targetY = tf.SparseTensor()

####Optimizing
logits = y_
logits3d = tf.stack(logits)
seqLengths = [20] * batch_size
cost = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
# CTCLoss op expects the reserved blank label to be the largest value! REALLY?

# if 1D:
tf.summary.scalar('cost', cost)
tf.summary.scalar('costs', costs)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(costs)
# prediction = y_

# Evaluate model
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# tf.scalar_summary('accuracy', accuracy)
# predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
# accuracy = tf.reduce_mean(tf.reduce_mean(logits))
# reduced_sum = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False))
# errorRate = reduced_sum / tf.to_float(tf.size(targetY.values))

steps = 9999999
session = tf.Session()
try:
	saver = tf.train.Saver(tf.global_variables())
except:
	saver = tf.train.Saver(tf.global_variables())
snapshot = "lstm_mfcc"
checkpoint = tf.train.latest_checkpoint(checkpoint_dir="checkpoints")
if checkpoint:
	print("LOADING checkpoint " + checkpoint + "")
	try: saver.restore(session, checkpoint)
	except: print("incompatible checkpoint")
try: session.run([tf.global_variables_initializer()])
except: session.run([tf.global_variables_initializer()])  # tf <12

# train
step = 0  # show first
try: summaries = tf.summary.merge_all()
except: summaries = tf.summary.merge_all()  # tf<12
try: summary_writer = tf.summary.FileWriter("logs", session.graph)  #
except: summary_writer = tf.summary.FileWriter("logs", session.graph)  # tf<12
while step < steps:
	batch_xs, batch_ys = next(batch)

	# tf.train.shuffle_batch_join(example_list, batch_size, capacity=min_queue_size + batch_size * 16, min_queue_size)
	# Fit training using batch data
	feed_dict = {x: batch_xs, y: batch_ys}
	# feed_dict = {inputX: batch_xs, targetIxs: batch_ys.indices, targetVals: batch_ys.values,targetShape: 20}
	# , seqLengths: batchSeqLengths
	loss, _ = session.run([costs, optimizer], feed_dict=feed_dict)
	if step % display_step == 0:
		seconds = int(time.time()) - start
		# Calculate batch accuracy, loss
		feed = {x: batch_xs, y: batch_ys}  # , keep_prob: 1., train_phase: False}
		acc, summary = session.run([accuracy, summaries], feed_dict=feed)
		# summary_writer.add_summary(summary, step) # only test summaries for smoother curve
		print("\rStep {:d} Loss={:.4f} Fit={:.1f}% Time={:d}s".format(step, loss, acc, seconds), end=' ')
		if str(loss) == "nan":
			print("\nLoss gradiant explosion, quitting!!!")  # restore!
			quit(0)
	# if step % test_step == 0: test(step)
	if step % save_step == 0 and step > 0:
		print("SAVING snapshot %s" % snapshot)
		saver.save(session, "checkpoints/" + snapshot + ".ckpt", step)
	step = step + 1
