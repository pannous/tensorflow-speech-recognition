"""A simple speech classifer AND autoencoder

it 2000 speech_entropy  9.46917 overal accuracy  0.9502   400
it 72600 cross_entropy  0.345358 overal accuracy  0.9714
it 73100 cross_entropy  nan overal accuracy  0.098  WHY NAN??

"""

# Import data
import sys
import tensorflow as tf
sess = tf.InteractiveSession()
import speech_data
speech = speech_data.read_data_sets("/data/speech/", one_hot=True)

width=256
height=256


def encode_model():
  n1=400
  n2=100
  n3=20
  # Create the model
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

  # Define loss and optimizer
  y_ = tf.placeholder("float", [None,10])
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
  return x,y,y_,speech_step,encod_step,train_step, accuracy



x,y,y_,speech_step,encod_step,train_step, accuracy=encode_model()

def eval(feed):
    print("iteration %d"%i, end=' ')
    print("cross_entropy ",cross_entropy.eval(feed), end=' ')
    print("encod_entropy ",encod_entropy.eval(feed), end=' ')
    print("speech_entropy ",speech_entropy.eval(feed), end=' ')
    print("overal accuracy ",accuracy.eval({x: speech.test.images, y_: speech.test.labels}))#, end='\r'WWWWAAA)


def train():
  tf.initialize_all_variables().run()
  print("Pretrain")
  for i in range(6000-1):
    batch_xs, batch_ys = speech.train.next_batch(100)
# ValueError: Cannot feed value of shape (100, 512, 512) for Tensor 'Placeholder:0', which has shape (Dimension(None), Dimension(65536))
    # todo a) flatten or b) add one dimension to all tensors involved
    feed = {x: batch_xs, y_: batch_ys}
    speech_step.run(feed) # better for encod_entropy too! (later)
    if(i%100==0):eval(feed)
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
    if(i%100==0):eval(feed)

if __name__ == '__main__':
    train()

def train_spectrogram_encoder():
  train()
