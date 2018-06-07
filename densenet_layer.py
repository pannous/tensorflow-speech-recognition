#!/usr/bin/python
import tensorflow as tf
import layer
import speech_data
from speech_data import Source,Target


learning_rate = 0.001
training_iters = 300000
batch_size = 64


# BASELINE toy net
def simple_dense(net): # best with lr ~0.001
	# type: (layer.net) -> None
	# net.dense(hidden=200,depth=8,dropout=False) # BETTER!!
	net.dense(400, activation=tf.nn.tanh)# 0.99 YAY
	# net.denseNet(40, depth=4)
	# net.classifier() # auto classes from labels
	return


def alex(net): # kinda
	# type: (layer.net) -> None
	print("Building Alex-net")
	net.reshape(shape=[-1, 64, 64, 1])  # Reshape input picture
	# net.batchnorm()
	net.conv([3, 3, 1, 64]) # 64 filters
	net.conv([3, 3, 64, 128])
	net.conv([3, 3, 128, 256])
	net.conv([3, 3, 256, 512])
	net.conv([3, 3, 512, 1024])
	net.dense(1024,activation=tf.nn.relu)
	net.dense(1024,activation=tf.nn.relu)


# Densely Connected Convolutional Networks https://arxiv.org/abs/1608.06993  # advanced ResNet
def denseConv(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	net.reshape(shape=[-1, 64, 64, 1])  # Reshape input picture
	net.buildDenseConv(nBlocks=1)  # increase nBlocks for real data
	net.classifier() # auto classes from labels


def denseNet(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	net.reshape(shape=[-1, 64, 64, 1])  # Reshape input picture
	net.fullDenseNet()
	net.classifier() # auto classes from labels


train_digits=True
if train_digits:
	width= height=64 # for pcm baby data
	batch=speech_data.spectro_batch_generator(1000,target=speech_data.Target.digits)
	classes=10 # digits
else:
	width=512 # for spoken_words overkill data
	classes=74 #
	batch=word_batch=speech_data.spectro_batch_generator(10, width, source_data=Source.WORD_SPECTROS, target=Target.first_letter)
	raise Exception("TODO")

X,Y=next(batch)

# CHOOSE MODEL ARCHITECTURE HERE:
# net = layer.net(simple_dense, data=batch, input_width=width, output_width=classes, learning_rate=0.01)
net = layer.net(simple_dense, data=batch, input_shape=(width,height), output_width=classes, learning_rate=0.01)
# net=layer.net(model=alex,input_shape=(width, height),output_width=10, learning_rate=learning_rate)
# net=layer.net(model=denseConv, input_shape=(width, height),output_width=10, learning_rate=learning_rate)

net.train(data=batch,batch_size=10,steps=500,dropout=0.6,display_step=1,test_step=1) # debug
# net.train(data=batch,batch_size=10,steps=5000,dropout=0.6,display_step=5,test_step=20) # test
# net.train(data=batch,batch_size=10,steps=5000,dropout=0.6,display_step=10,test_step=100) # run

# net.predict() # nil=random
# net.generate(3)  # nil=random

print ("Now try switching between model architectures in line 68-71")
