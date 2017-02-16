#!/usr/bin/env python
import numpy as np
import tensorflow as tf

def target_list_to_sparse_tensor(targetList):
		'''make tensorflow SparseTensor from list of targets, with each element
			 in the list being a list or array with the values of the target sequence
			 (e.g., the integer values of a character map for an ASR target string)
			 See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
			 for example of SparseTensor format'''
		indices = []
		vals = []
		for tI, target in enumerate(targetList):
				for seqI, val in enumerate(target):
						indices.append([tI, seqI])
						vals.append(val)
		shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
		return (np.array(indices), np.array(vals), np.array(shape))

def test_edit_distance():
		graph = tf.Graph()
		with graph.as_default():
				truth = tf.sparse_placeholder(tf.int32)
				hyp = tf.sparse_placeholder(tf.int32)
				editDist = tf.edit_distance(hyp, truth, normalize=False)

		with tf.Session(graph=graph) as session:
				truthTest = sparse_tensor_feed([[0,1,2], [0,1,2,3,4]])
				hypTest = sparse_tensor_feed([[3,4,5], [0,1,2,2]])
				feedDict = {truth: truthTest, hyp: hypTest}
				dist = session.run([editDist], feed_dict=feedDict)
				print(dist)

def data_lists_to_batches(inputList, targetList, batchSize,swap_axes=True):
		'''Takes a list of input matrices and a list of target arrays and returns
			 a list of batches, with each batch being a 3-element tuple of inputs,
			 targets, and sequence lengths.
			 inputList: list of 2-d numpy arrays with dimensions nFeatures x timesteps
			 targetList: list of 1-d arrays or lists of ints
			 batchSize: int indicating number of inputs/targets per batch
			 returns: dataBatches: list of batch data tuples, where each batch tuple (inputs, targets, seqLengths) consists of
										inputs = 3-d array w/ shape nTimeSteps x batchSize x nFeatures
										targets = tuple required as input for SparseTensor
										seqLengths = 1-d array with int number of timesteps for each sample in batch
								maxSteps: maximum number of time steps across all samples'''

		assert len(inputList) == len(targetList), "%d!=%d"%(len(inputList),len(targetList))
		nFeatures = inputList[0].shape[0]
		maxSteps = 0
		for inp in inputList:
			maxSteps = max(maxSteps, inp.shape[1])
		print("nFeatures: %d    maxSteps: %d  "%(nFeatures,maxSteps))

		randIxs = np.random.permutation(len(inputList))
		start, end = (0, batchSize)
		dataBatches = []

		while end <= len(inputList):
				batchSeqLengths = np.zeros(batchSize)
				for batchI, origI in enumerate(randIxs[start:end]):
						batchSeqLengths[batchI] = inputList[origI].shape[-1]
				batchInputs = np.zeros((maxSteps, batchSize, nFeatures))
				batchTargetList = []
				for batchI, origI in enumerate(randIxs[start:end]):
						orig = inputList[origI]
						# print("orig shape: "+ str( orig.shape))
						padSecs = maxSteps - orig.shape[1]
						# padFeatures = maxFeatures - orig.shape[0]
						padded = np.pad(orig.T, ((0, padSecs), (0, 0)), 'constant', constant_values=0)
						# print("transposed + padded "+str(padded.shape))
						batchInputs[:,batchI,:] = padded
						batchTargetList.append(targetList[origI])
				dataBatches.append((batchInputs, target_list_to_sparse_tensor(batchTargetList),
													batchSeqLengths))
				start += batchSize
				end += batchSize
		return (dataBatches, maxSteps)

def load_batched_data(specPath, targetPath, batchSize, match=True):
		import os
		'''returns 4-element tuple: batched data (list), max # of time steps (int),
			 total number of samples (int) and number of classes incl. noc (int)'''
		print("load_batched_data samples from " + specPath)
		print("load_batched_data lables  from " + targetPath)
		a = [np.load(os.path.join(specPath, fn)) for fn in os.listdir(specPath)]
		if match:b = [np.load(os.path.join(targetPath, fn)) for fn in os.listdir(specPath)]
		else: b = [np.load(os.path.join(targetPath, fn)) for fn in os.listdir(targetPath)]
		nSamples = len(os.listdir(specPath))
		nLables = len(os.listdir(targetPath))
		nClasses = -2359817587589759875328957 # doesn't matter
		print("#samples=?=#lables : %d=?=%d => %d=?=%d"%(nSamples,nLables,len(a),len(b)))
		return data_lists_to_batches(a, b, batchSize) + (nSamples, nClasses)
