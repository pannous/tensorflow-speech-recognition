#!/usr/bin/env python
#!/usr/bin/env python
'''
Example of a single-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict character sequences from nFeatures x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  This is test code to run on the
8-item data set in the "sample_data" directory, for those without access to TIMIT.

Original Author: Jon Rein
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
import re

from bdlstm_utils import load_batched_data

INPUT_PATH = '/data/ctc/sample_data/mfcc'  # directory of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = '/data/ctc/sample_data/char_y/'  # directory of nCharacters 1-D array .npy files
# HUH?? >>> np.load("/data/ctc/sample_data/char_y/0.npy")
# array([19, 15, 26, 2, 0, 18, 22, 18, 12, 2, 0, 18, 4, 16, 1, 0, 8,
#        26, 17, 26, 14, 12, 2, 0, 8, 22, 13, 0, 15, 4, 14, 12, 19, 15,
#        12, 0, 15, 5, 22, 15, 19, 25, 4, 0, 15, 26, 14, 12, 26, 1, 0,
#        4, 14, 26, 0, 4, 14, 26, 0, 26, 22, 25, 5, 12, 0, 3, 4, 22,
#        14, 12, 0, 12, 21, 4, 0, 12, 21, 4], dtype=uint8)
# >> > map(lambda x: chr(x + 64), _)
# ['S', 'O', 'Z', 'B', '@', 'R', 'V', 'R', 'L', 'B', '@', 'R', 'D', 'P', 'A', '@', 'H', 'Z', 'Q', 'Z', 'N', 'L', 'B', '@',
#  'H', 'V', 'M', '@', 'O', 'D', 'N', 'L', 'S', 'O', 'L', '@', 'O', 'E', 'V', 'O', 'S', 'Y', 'D', '@', 'O', 'Z', 'N', 'L',
#  'Z', 'A', '@', 'D', 'N', 'Z', '@', 'D', 'N', 'Z', '@', 'Z', 'V', 'Y', 'E', 'L', '@', 'C', 'D', 'V', 'N', 'L', '@', 'L',
#  'U', 'D', '@', 'L', 'U', 'D']
# our_data=True
our_data = False
if our_data:
	print("Using our data")
	INPUT_PATH = 'data/number/mfcc'  # directory of MFCC nFeatures x nFrames 2-D array .npy files
	TARGET_PATH = 'data/number/chars/'  # directory of nCharacters 1-D array .npy files
	# we have 0.npy : ~ zEeRo : array([31, 26, 17,  9, 25, 18, 15, 23, 14,  0, ... ]) should be fine?

####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 300
Size = 4

####Network Parameters

nFeatures = 26  # 12 MFCC coefficients + energy, and derivatives
nClasses = 28  # 27 characters, plus the "blank" for CTC
if our_data:
	nFeatures = 26  # 20 MFCC coefficients + ^^ WHERE DID YOU GET THOSE?
	nClasses = 32 # forgot why ;)

nHidden = 128
####Load data
print('Loading data')
edData, maxTimeSteps, totalN, _ = load_batched_data(INPUT_PATH, TARGET_PATH, Size,match=our_data)

####Define graph
print('Defining graph')
print('This takes forever + some')
print('Impossible to debug')
print('It is slower than any other tensorflow graph, why?')

graph = tf.Graph()
with graph.as_default():
	####NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow

	####Graph input
	inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, Size, nFeatures))
	# Prep input data to fit requirements of rnn.bidirectional_rnn
	#  Reshape to 2-D tensor (nTimeSteps*Size, nfeatures)
	inputXrs = tf.reshape(inputX, [-1, nFeatures])
	#  Split to get a list of 'n_steps' tensors of shape (_size, n_hidden)
	inputList = tf.split(axis=0, num_or_size_splits=maxTimeSteps, value=inputXrs)
	targetIxs = tf.placeholder(tf.int64)
	targetVals = tf.placeholder(tf.int32)
	targetShape = tf.placeholder(tf.int64)
	targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
	seqLengths = tf.placeholder(tf.int32, shape=Size)

	####Weights & biases
	stddev = np.sqrt(2.0 / (2 * nHidden))
	truncated_normal = tf.truncated_normal([2, nHidden], stddev=stddev)
	weightsOutH1 = tf.Variable(truncated_normal)
	biasesOutH1 = tf.Variable(tf.zeros([nHidden]))
	weightsOutH2 = tf.Variable(truncated_normal)
	biasesOutH2 = tf.Variable(tf.zeros([nHidden]))
	half_normal = tf.truncated_normal([nHidden, nClasses], stddev=np.sqrt(2.0 / nHidden))
	weightsClasses = tf.Variable(half_normal)
	biasesClasses = tf.Variable(tf.zeros([nClasses]))

	####Network
	forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
	backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
	print("building bidirectional_rnn ... SLOW!!!")
	fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32, scope='BDLSTM_H1')
	print("done building rnn")
	print("building fbH1rs ")
	fbH1rs = [tf.reshape(t, [Size, 2, nHidden]) for t in fbH1]
	print("building outH1 ")
	outH1 = [tf.reduce_sum(tf.multiply(t, weightsOutH1), axis=1) + biasesOutH1 for t in fbH1rs]
	print("building logits ")
	logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]
	print("len(outH1) %d"% len(outH1))
	####Optimizing
	print("building loss")
	logits3d = tf.stack(logits)
	loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
	out = tf.identity(loss, 'ctc_loss_mean')
	optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)

	####Evaluating
	print("building Evaluation")
	logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seqLengths[0], 1])
	predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
	reduced_sum = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False))
	errorRate = reduced_sum / tf.to_float(tf.size(targetY.values))

	check_op = tf.add_check_numerics_ops()
print("done building graph")

####Run session
with tf.Session(graph=graph) as session:
	try: merged = tf.summary.merge_all()
	except: merged = tf.summary.merge_all()
	try:writer = tf.summary.FileWriter("/tmp/basic_new", session.graph)
	except: writer = tf.summary.FileWriter("/tmp/basic_new", session.graph)
	try:saver = tf.train.Saver()  # defaults to saving all variables
	except:
		print("tf.train.Saver() broken in tensorflow 0.12")
		saver = tf.train.Saver(tf.global_variables())# WTF stupid API breaking
	ckpt = tf.train.get_checkpoint_state('./checkpoints')

	start = 0
	if ckpt and ckpt.model_checkpoint_path:
		p = re.compile('\./checkpoints/model\.ckpt-([0-9]+)')
		m = p.match(ckpt.model_checkpoint_path)
		try:start = int(m.group(1))
		except:pass
	if saver and start > 0:
		# Restore variables from disk.
		saver.restore(session, "./checkpoints/model.ckpt-%d" % start)
		print("Model %d restored." % start)
	else:
		print('Initializing')
		try: session.run(tf.global_variables_initializer())
		except:session.run(tf.global_variables_initializer())
	for epoch in range(nEpochs):
		print('Epoch', epoch + 1, '...')
		errors = np.zeros(len(edData))
		RandIxs = np.random.permutation(len(edData))  # randomize  order
		for Nr, OrigI in enumerate(RandIxs):
			Inputs, TargetSparse, SeqLengths = edData[OrigI]
			indices, values, shape = TargetSparse
			feedDict = {inputX: Inputs, targetIxs: indices, targetVals: values, targetShape: shape, seqLengths: SeqLengths}
			_, l, er, lmt, ok = session.run([optimizer, loss, errorRate, logitsMaxTest, check_op], feed_dict=feedDict)
			print(np.unique(lmt))
			# print unique argmax values of first sample in ; should be blank for a while, then spit out target values
			if (Nr % 1) == 0:
				print('Mini', Nr, '/', OrigI, 'loss:', l)
				print('Mini', Nr, '/', OrigI, 'error rate:', er)
			errors[Nr] = er * len(SeqLengths)
		epochErrorRate = errors.sum() / totalN
		print('Epoch', epoch + 1, 'error rate:', epochErrorRate)
		if saver:saver.save(session, 'checkpoints/model.ckpt', global_step=epoch + 1)
	print('Learning finished')

