#!/usr/bin/env python
#!/usr/bin/python
import tensorflow as tf
import layer
import speech_data
from speech_data import Source,Target
from layer import net


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


def denseConv(net):
	# type: (layer.net) -> None
	print("Building dense-net")
	net.reshape(shape=[-1, 64, 64, 1])  # Reshape input picture
	net.buildDenseConv()
	net.classifier() # auto classes from labels


# width=64 # for pcm baby data
# batch=speech_data.spectro_batch_generator(1000,target=speech_data.Target.digits)
# classes=10

width=512 # for spoken_words overkill data
classes=74 #
batch=word_batch=speech_data.spectro_batch_generator(10,width,source_data=Source.SPOKEN_WORDS, target=Target.first_letter)
X,Y=next(batch)

# CHOSE MODEL ARCHITECTURE HERE:
net=layer.net(simple_dense, width*width, classes, learning_rate=0.01)
# net=layer.net(model=alex,input_width=64*64,output_width=10, learning_rate=0.001)
# net=layer.net(model=denseConv,input_width=64*64,output_width=10, learning_rate=0.001)

net.train(data=batch,batch_size=10,steps=500,dropout=0.6,display_step=1,test_step=1) # debug
# net.train(data=batch,batch_size=10,steps=5000,dropout=0.6,display_step=5,test_step=20) # test
# net.train(data=batch,batch_size=10,steps=5000,dropout=0.6,display_step=10,test_step=100) # run

# net.predict() # nil=random
# net.generate(3)  # nil=random
