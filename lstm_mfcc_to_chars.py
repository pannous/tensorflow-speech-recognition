#!/usr/bin/env python
#!/usr/local/bin/python
from __future__ import print_function
import numpy as np
import tensorflow as tf
import layer
import speech_data
from speech_data import Source,Target
from layer import net
import time
start=int(time.time())
display_step = 1
test_step = 10
save_step = 100
learning_rate = 0.0001
# 0.0001 Step 300 Loss= 1.976625 Accuracy= 0.250 Time= 303s
# Step 24261 Loss= 0.011786 Accuracy= 1.000 Time= 33762s takes time but works

training_iters = 300000 #steps
batch_size = 64

width=features=20 # mfcc features
height=max_length=80 # (max) length of utterance
classes=10 # digits

keep_prob=dropout=0.7

batch = speech_data.mfcc_batch_generator(batch_size,target=Target.digits) #
X,Y=next(batch)
# print(Y)
print(np.array(Y).shape)

# inputs=tf.placeholder(tf.float32, shape=(batch_size,max_length,features))
x=inputs=tf.placeholder(tf.float32, shape=(batch_size,features,max_length))
# inputs = tf.transpose(inputs, [0, 2, 1]) #  inputs must be a `Tensor` of shape: `[batch_size, max_time, ...]`
inputs = tf.transpose(inputs, [2, 0, 1]) # [max_time, batch_size, features] to split:
# Split data because rnn cell needs a list of inputs for the RNN inner loop
inputs = tf.split(axis=0, num_or_size_splits=max_length, value=inputs)  # n_steps * (batch_size, features)

num_hidden = 100 #features
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
# rnn=tf.nn.rnn(cell,inputs)
# rnn=tf.nn.dynamic_rnn(cell,inputs)
# manual:

state = cell.zero_state(batch_size, dtype=tf.float32)
if "manual" == 0:
	outputs = []
	for input_ in inputs:
		input_= tf.reshape(input_, [batch_size,features])
		output, state = cell(input_, state)
		outputs.append(output)
	y_=output
else:
	# inputs = tf.reshape(inputs, [-1, features])
	inputs=[tf.reshape(input_, [batch_size,features]) for input_ in inputs]
	outputs, states = tf.nn.rnn(cell, inputs, initial_state=state)
	# only last output as target for now
	y_=outputs[-1]

# dense
weights = tf.Variable(tf.random_uniform([num_hidden, classes], minval=-1. / width, maxval=1. / width), name="weights_dense")
bias = tf.Variable(tf.random_uniform([classes], minval=-1. / width, maxval=1. / width), name="bias_dense")
y_ = tf.matmul(y_, weights, name='dense' ) + bias

# optimize
# if use_word: y=target=tf.placeholder(tf.float32, shape=(batch_size,(None,32))) # -> seq2seq!
y=target=tf.placeholder(tf.float32, shape=(batch_size,classes))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_,labels=y),name="cost")  # prediction, target
tf.summary.scalar('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
prediction = y_
# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

steps = 9999999
session=tf.Session()
saver = tf.train.Saver(tf.global_variables())
snapshot = "lstm_mfcc"
checkpoint = tf.train.latest_checkpoint(checkpoint_dir="checkpoints")
if checkpoint:
	print("LOADING " + checkpoint + " !!!")
	try:saver.restore(session, checkpoint)
	except: print("incompatible checkpoint")
try: session.run([tf.global_variables_initializer()])
except: session.run([tf.global_variables_initializer()])


#train
step = 0  # show first
summaries = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("logs", session.graph)  #
while step < steps:
	batch_xs, batch_ys = next(batch)
	# tf.train.shuffle_batch_join(example_list, batch_size, capacity=min_queue_size + batch_size * 16, min_queue_size)
	# Fit training using batch data
	feed_dict = {x: batch_xs, y: batch_ys}
	loss, _ = session.run([cost, optimizer], feed_dict=feed_dict)
	if step % display_step == 0:
		seconds = int(time.time()) - start
		# Calculate batch accuracy, loss
		feed = {x: batch_xs, y: batch_ys} #, keep_prob: 1., train_phase: False}
		acc, summary = session.run([accuracy, summaries], feed_dict=feed)
		# summary_writer.add_summary(summary, step) # only test summaries for smoother curve
		print("\rStep {:d} Loss= {:.6f} Fit= {:.3f} Time= {:d}s".format(step, loss, acc, seconds), end=' ')
		if str(loss) == "nan":
			print("\nLoss gradiant explosion, quitting!!!")  # restore!
			quit(0)
	# if step % test_step == 0: test(step)
	if step % save_step == 0 and step > 0:
		print("SAVING snapshot %s" % snapshot)
		saver.save(session, "checkpoints/" + snapshot + ".ckpt", step)
	step = step +1
