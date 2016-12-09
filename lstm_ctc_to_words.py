'''
Example of a single-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict phoneme sequences from nFeatures x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  This is basically a recreation of an experiment
on the TIMIT data set from chapter 7 of Alex Graves's book (Graves, Alex. Supervised Sequence
Labelling with Recurrent Neural Networks, volume 385 of Studies in Computational Intelligence.
Springer, 2012.), minus the early stopping.

Author: Jon Rein
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
from bdlstm_utils import load_batched_data

INPUT_PATH = '/data/ctc/mfcc/'  # directory of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = '/data/ctc/char_y/'  # directory of nPhonemes 1-D array .npy files


####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 120
batchSize = 128

####Network Parameters
nFeatures = 26  # 12 MFCC coefficients + energy, and derivatives
nHidden = 128
# nClasses = 40  # 39 phonemes, plus the "blank" for CTC

####Load data
print('Loading data')
batchedData, maxTimeSteps, totalN, nClasses = load_batched_data(INPUT_PATH, TARGET_PATH, batchSize)

####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():
	####NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow

	####Graph input
	inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, batchSize, nFeatures))
	# Prep input data to fit requirements of rnn.bidirectional_rnn
	#  Reshape to 2-D tensor (nTimeSteps*batchSize, nfeatures)
	inputXrs = tf.reshape(inputX, [-1, nFeatures])
	#  Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
	inputList = tf.split(0, maxTimeSteps, inputXrs)
	targetIxs = tf.placeholder(tf.int64)
	targetVals = tf.placeholder(tf.int32)
	targetShape = tf.placeholder(tf.int64)
	targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
	seqLengths = tf.placeholder(tf.int32, shape=(batchSize))

	####Weights & biases
	normal = tf.truncated_normal([2, nHidden], stddev=np.sqrt(2.0 / (2 * nHidden)))
	weightsOutH1 = tf.Variable(normal)
	biasesOutH1 = tf.Variable(tf.zeros([nHidden]))
	weightsOutH2 = tf.Variable(normal)
	biasesOutH2 = tf.Variable(tf.zeros([nHidden]))
	half_normal = tf.truncated_normal([nHidden, nClasses], stddev=np.sqrt(2.0 / nHidden))
	weightsClasses = tf.Variable(half_normal)
	biasesClasses = tf.Variable(tf.zeros([nClasses]))

	####Network
	forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
	backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
	fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32, scope='BDLSTM_H1')
	fbH1rs = [tf.reshape(t, [batchSize, 2, nHidden]) for t in fbH1]
	outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]

	logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

	####Optimizing
	logits3d = tf.pack(logits)
	loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
	optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)

	####Evaluating
	logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seqLengths[0], 1])
	predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
	reduced_sum = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False))
	errorRate = reduced_sum / tf.to_float(tf.size(targetY.values))

####Run session
with tf.Session(graph=graph) as session:
	print('Initializing')
	tf.initialize_all_variables().run()
	for epoch in range(nEpochs):
		print('Epoch', epoch + 1, '...')
		batchErrors = np.zeros(len(batchedData))
		batchRandIxs = np.random.permutation(len(batchedData))  # randomize batch order
		for batch, batchOrigI in enumerate(batchRandIxs):
			batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
			batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
			feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
			            targetShape: batchTargetShape, seqLengths: batchSeqLengths}
			_, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
			print(np.unique(lmt))
			# print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
			if (batch % 1) == 0:
				print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
				print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
			batchErrors[batch] = er * len(batchSeqLengths)
		epochErrorRate = batchErrors.sum() / totalN
		print('Epoch', epoch + 1, 'error rate:', epochErrorRate)
