#!/usr/bin/env python
#!/usr/local/bin/python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import speech_data

sess = tf.InteractiveSession()

batch_size=100
width=height=64
dim=width*height
# # Create the classifier model
x2 = tf.placeholder("float", [batch_size, width, height], name='image_batch')  # None~batch_size
x = tf.reshape(x2, [batch_size, dim])  # flatten
W = tf.Variable(tf.zeros([dim,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
# # Define loss and optimizer
y_ = tf.placeholder("float", [batch_size,10],name='label_batch')
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# GAN generative adversarial network:

# Create the generator model
y0 = y_ # share tf.placeholder("float", [10],name="seed")
Wg = tf.Variable(tf.zeros([10,dim]),name='W_generator')
xg=generated_x = tf.matmul(y0, Wg)

# Create the discriminator model
x0 = tf.Variable(tf.zeros([batch_size,dim]))
discriminate = tf.assign(x0,x) # feed real data batch
generate = tf.assign(x0,generated_x) # feed generated data batch

Wd = tf.Variable(tf.zeros([dim,1]))
bd = tf.Variable(tf.zeros([1]))
verdict = tf.sigmoid( tf.matmul(x, Wd) + bd)
# Define loss and optimizer
verdict_ = tf.placeholder("float", [batch_size], name='verdict') # is this sample artificial '0' or real '1' ?

lam=0.0000001
# lam=0.01
# discriminator_entropy = -tf.reduce_sum(verdict_ * tf.log(verdict))
discriminator_entropy = tf.reduce_mean(tf.square(verdict_-verdict))
generator_entropy = tf.reduce_mean(tf.square(x-generated_x))
# cross_entropy = tf.reduce_sum(abs(y_-y))

gan_entropy = discriminator_entropy + lam*generator_entropy
gan_step  = tf.train.AdamOptimizer(learning_rate=0.04).minimize(gan_entropy) # 0.04=good #ANY VALUE WORKS!! WOW


negative=[0]*batch_size # input was fake
positive=[1]*batch_size # input was real

def check_accuracy():
  # Test trained model
  prediction=tf.argmax(y,1)
  probability=(y) #tf.div(y,tf.reduce_sum(y,0))
  correct_prediction = tf.equal(prediction, tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  batch_xs, batch_ys = next(batch)
  feed_dict = {x2: batch_xs, y_: batch_ys}
  best,p,a,verdict1= sess.run([prediction,probability,accuracy,verdict],feed_dict)
  # print(best,a,list(map(lambda x:round(x,3),p[0])))
  print("overal accuracy ",a)


batch=speech_data.spectro_batch(batch_size)

draw=0

# Train
tf.global_variables_initializer().run()
steps=30000
e=0
for i in range(steps):
  batch_xs, batch_ys = next(batch)
  # use x2 for matrix, x for flattened data
  train_step.run({x2: batch_xs, y_: batch_ys})  # classical classifier
  _, _, verdict1 = sess.run([discriminate, gan_step, verdict],
                            {x2: batch_xs, y_: batch_ys, verdict_: positive})  # true examples
  sampled, _, loss, verdict0 = sess.run([generate, gan_step, generator_entropy, verdict],
                                        {x2: batch_xs, y_: batch_ys, verdict_: negative})  # generated samples
  e+=loss
  if(i%10==0):
    print("loss ",e)
    e=0

  if (i % 100 == 0):
    # print("Fool factor 0:  how often has it been fooled: %d %%"%(sum(verdict0)))
    # print("Fool factor 1:  how often did it identify true samples %d %%"%(sum(verdict1)))
    print("identified true samples %d%%  fooled: %d%%" % (sum(verdict1),sum(verdict0)))
    # imgs=np.reshape(batch_xs,(batch_size,28,28))[0]
    check_accuracy()
    if(draw):
      imgs=np.reshape(sampled,(batch_size,64,64))[0]
      plt.matshow(imgs,fignum=1)
      # plt.matshow(sampled,fignum=1)
      plt.draw()
      plt.pause(0.01)


sampled = sess.run(generated_x,{y_: [[0,0,0,3,0,0,0,0,0,0]]*batch_size})  # generated samples
imgs = np.reshape(sampled, (batch_size, 64, 64))[0]
plt.matshow(imgs, fignum=2)
plt.show()
