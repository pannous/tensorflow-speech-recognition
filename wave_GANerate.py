#!/usr/bin/env python
#!/usr/bin/python 
# PYTHONUNBUFFERED=1
"""A simple GAN network and classifer.
"""
# from __future__ import print_function
# Import data
import matplotlib.pyplot as plt
import pyaudio
import wave
import sys
import skimage
from skimage.transform import resize, rescale
import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()

import speech_data

number_of_classes=10 # 10 digits
input_width=4096*2 # CHUNK*6 vs width*height
batch_size=10

# Create the classifier model
x = tf.placeholder("float", [batch_size, input_width],name='wave_batch') # None~batch_size
x0 = tf.Variable(tf.zeros([batch_size,input_width]),name='classifier_input')
hidden1size=64 #number_of_classes
W1 = tf.Variable(tf.truncated_normal([input_width,hidden1size]))
b1 = tf.Variable(tf.zeros([hidden1size]))
y1 = tf.nn.softmax(tf.matmul(x,W1) + b1)
y1 = tf.nn.dropout(y1,0.5)
W = tf.Variable(tf.truncated_normal([hidden1size,number_of_classes]))
b = tf.Variable(tf.zeros([number_of_classes]))
y = tf.nn.softmax(tf.matmul(y1,W) + b)
# Define loss and optimizer
y_ = tf.placeholder("float", [batch_size,number_of_classes],name='label_batch')
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# cross_entropy = tf.reduce_sum(abs(y_-y))
# cross_entropy = tf.reduce_sum(tf.square(y_-y))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
assign_batch = tf.assign(x0,x) # feed real data batch (vs gan_assign later)

# GAN generative adversarial network:

# Create the discriminator model
Wd = tf.Variable(tf.zeros([input_width,1]))
bd = tf.Variable(tf.zeros([1]))
verdict = tf.sigmoid( tf.matmul(x0, Wd) + bd)
verdict_ = tf.placeholder("float", [batch_size], name='verdict') # is this sample artificial '0' or real '1' ?
discriminator_entropy = tf.reduce_mean(tf.square(verdict_-verdict))
# discriminator_entropy = -tf.reduce_sum(verdict_ * tf.log(verdict))

# Create the generator model
y0 = y_ # share tf.placeholder("float", [10],name="seed")
Wg = tf.Variable(tf.zeros([10,input_width]),name='W_generator')
xg=generated_x = tf.matmul(y0, Wg)
generator_entropy = tf.reduce_mean(tf.square(x-generated_x))

#  evaluate and optimize the GAN's generator and discriminator
# lam=0.0000001
lam=0.01
gan_assign= tf.assign(x0,generated_x) # feed generated data batch into classifier
gan_entropy = discriminator_entropy + lam*generator_entropy
# gan_entropy = lam*cross_entropy + generator_entropy
gan_step  = tf.train.AdamOptimizer(learning_rate=0.04).minimize(gan_entropy) # 0.04=good #ANY VALUE WORKS!! WOW



def play_pcm(data):
  print("play_pcm")
  # f = wave.open(r"./test.wav", "rb")
  audio = pyaudio.PyAudio()
  # format=pyaudio.paFloat32
  format=pyaudio.paInt8
  # format=audio.get_format_from_width(f.getsampwidth())
  # out_stream = audio.open( format=format,channels = f.getnchannels(), rate=f.getframerate(), output= True)
  out_stream = audio.open( format=format,channels = 1, rate=48000, output= True)
  out_stream.start_stream()
  out_stream.write(data)


# Train
tf.global_variables_initializer().run()
steps=3000#000
batch=speech_data.wave_batch_generator(target=speech_data.Target.digits)
negative=[0]*batch_size # input was fake
positive=[1]*batch_size # input was real
# print(next(batch))
err=0
batch_xs, batch_ys = next(batch) #  keep constant for overfitting
for i in range(steps):
  # batch_xs, batch_ys = next(batch)
  # batch_xs1 = np.reshape(batch_xs,[batch_size,width*height])
  feed_dict={x: batch_xs, y_: batch_ys}
  _,loss=sess.run([train_step,cross_entropy],feed_dict) # classical classifier
  # _,_,loss=sess.run([assign_batch,train_step,cross_entropy],feed_dict) # classical classifier
  # feed_dict[verdict_]=positive # true examples
  # _, _, verdict1 =sess.run([assign_batch,gan_step,verdict],feed_dict)
  #
  # feed_dict[verdict_]=negative # generated samples
  # sampled, _ , _, verdict0 =sess.run([generated_x,gan_step,gan_assign,verdict], feed_dict)
  # sampled,_,_,loss=sess.run([generated_x,gan_step,gan_assign,gan_entropy],feed_dict) # gan classifier
  err+=loss
  if(i%20==0):
    print "%d loss %f\r"%(i,err),
    sys.stdout.flush()
    # print("%d loss %f\r"%(i,err), end='')#,flush=True) WTF PY3 
    err=0

  if(i%250==1):
    # play_pcm(sampled)
    # check_accuracy()
    # Test trained model
    prediction=tf.argmax(y,1)
    probability=(y) #tf.div(y,tf.reduce_sum(y,0))
    correct_prediction = tf.equal(prediction, tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # if not overfit: batch_xs, batch_ys = next(batch)
    feed_dict={x: batch_xs, y_: batch_ys}
    best,p,a,verdict1= sess.run([prediction,probability,accuracy,verdict],feed_dict)
    # print(best,a,list(map(lambda x:round(x,3),p[0])))
    print("\noveral accuracy %f"%a)



print("FINAL TEST:")
sampled = sess.run(generated_x,{y_: [[0,0,0,3,0,0,0,0,0,0]]*batch_size})  # generated samples
play_pcm(sampled)
