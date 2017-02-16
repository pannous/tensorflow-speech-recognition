#!/usr/bin/env python
"""A simple speech classifer AND autoencoder

1st step : classify+encode the spectogram of 10 spoken digits into one-hot-vector + 'flavor'-vector of size 10
The 'flavor' vector is meant to hold voice characteristics orthogonal to the 'number' information: gender, speed, pitch, ...

INPUT: spectogram
ENCODED: classifed digit + flavor
OUTPUT: reconstructed spectogram

iteration 2000 speech_entropy  9.46917 overal accuracy:  0.9502
iteration 72600 cross_entropy  0.345358 overal accuracy:  0.9714
iteration 73100 cross_entropy  nan overal accuracy  0.098  WHY NAN??

"""

import sys
import tensorflow as tf
sess = tf.InteractiveSession()
import speech_data
speech = speech_data.read_data_sets("data/", one_hot=True)

# width=256
# height=256
width=512
height=512

n1=400
n2=100
n3=20

print("Creating the model")

# input: spectogram (None: batch of variable size)
x = tf.placeholder("float", [None, width * height])

# l_rate = tf.placeholder("float", [1])# 0.01
l_rate = tf.Variable(0.003)

W1 = tf.Variable(tf.random_uniform([width * height,400],maxval=0.0001))
b1 = tf.Variable(tf.random_uniform([400]))
W2 = tf.Variable(tf.random_uniform([400,100],maxval=0.001))
b2 = tf.Variable(tf.random_uniform([100]))
W3 = tf.Variable(tf.random_uniform([100,n3],maxval=0.01))
b3 = tf.Variable(tf.zeros([n3]))
h= tf.nn.tanh( tf.matmul(x,W1)+b1) #
h= tf.nn.dropout(h,keep_prob=.5) # THAT!
h2= tf.nn.tanh( tf.matmul(h,W2)+b2) #
# _y = tf.matmul(h,W2) + b2 # 10 Numbers + 10 'styles'
_y = tf.matmul(h2,W3) + b3 # 10 Numbers + 10 'styles'
# print("_y ",tf.rank(_y))
if n3==20:
  y = tf.nn.softmax(tf.slice(_y,[0,0],[-1,10])) #  softmax of all batches(-1) only on the numbers(10)
else:
  y = tf.nn.softmax(_y)

e2= tf.matmul(_y,tf.transpose(W3)) #+b2
e2=tf.nn.tanh(e2)
e1= tf.matmul(e2,tf.transpose(W2)) #+b2  _y
e1=tf.nn.tanh(e1)

x_=x_reconstructed= tf.nn.sigmoid(tf.matmul(e1,tf.transpose(W1)))
# measure reconstructed signal against input:
y_ = tf.placeholder("float", [None,10])

# Define loss and optimizer
speech_entropy = -tf.reduce_sum(y_*tf.log(y))

# encod_entropy = -tf.reduce_sum(x_*tf.log(x))
encod_entropy = tf.sqrt(tf.reduce_mean(tf.square(x - x_)))
# encod_entropy = tf.reduce_mean(tf.square(x - x_))

cross_entropy = encod_entropy * speech_entropy
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))

speech_step = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(speech_entropy)
encod_step = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(encod_entropy)
train_step = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_,1)), "float"))

def flatten(matrix):
  # print("flattening %d"
  # return itertools.chain.from_iterable(matrix)
  return [item for vector in matrix for item in vector]


test_images=[flatten(matrix) for matrix in speech.test.images]

def eval(feed):
    print("cross_entropy ",cross_entropy.eval(feed))#, end=' ')
    print("encod_entropy ",encod_entropy.eval(feed))#, end=' ')
    print("speech_entropy ",speech_entropy.eval(feed))#, end=' ')
    print("overal accuracy ",accuracy.eval({x: test_images, y_: speech.test.labels}))#, end='\r'WWWWAAA)


def train_spectrogram_encoder():
  tf.global_variables_initializer().run()
  print("Pretrain")
  for i in range(6000-1):
    batch_xs, batch_ys = speech.train.next_batch(100)
    # WTF, tensorflow can't do 3D tensor operations?
    # https://github.com/tensorflow/tensorflow/issues/406 =>
    batch_xs=[flatten(matrix) for matrix in batch_xs]
    #  you have to reshape to flat/matrix data? why didn't they call it matrixflow?
    feed = {x: batch_xs, y_: batch_ys}
    speech_step.run(feed) # better for encod_entropy too! (later)
    if(i%100==0):
        print("iteration %d"%i)#, end=' ')
        eval(feed)
    if((i+1)%7000==0):
      print("l_rate*=0.1")
      sess.run(tf.assign(l_rate,l_rate*0.1))
  print("Train")
  for i in range(100000):
    batch_xs, batch_ys = speech.train.next_batch(100)
    feed = {x: batch_xs, y_: batch_ys}
    if((i+1)%9000==0):sess.run(tf.assign(l_rate,l_rate*0.3))
    encod_step.run(feed) # alternating!
    speech_step.run(feed)
    train_step.run(feed)
    if(i%100==0):
      print("iteration %d"%i)#, end=' ')
      eval(feed)

if __name__ == '__main__':
  train_spectrogram_encoder()

