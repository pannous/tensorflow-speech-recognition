"""A simple speech classifer AND autoencoder
"""
from __future__ import print_function
import pyaudio
import wave
import numpy
import numpy as np
import pymouse
import pygame
import math

# pygame.init()
# pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=4096)
pygame.mixer.init(frequency=8000, size=-16, channels=1, buffer=4096)
mouse = pymouse.PyMouse() # as sine input

# import keyboard_listener
# keyboard_listener.listen_keyboard(lambda x:print("x"+x))
#define stream chunk
chunk = 1024
CHUNK = 4096
# CHUNK = 9192
# length=512
length = 1024
volume = 0.2     # range [0.0, 1.0]
# length=2048

#open a wav format music
# import wave
# w = wave.open('/usr/share/sounds/ekiga/voicemail.wav', 'r')
# for i in range(w.getnframes()):
#   frame = w.readframes(i)
#   print frame

# import sounddevice as sd
# CHANNELS = 2
# RECORD_SECONDS = 5
# myrecording = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=CHANNELS, blocking=True, dtype='float32')

#instantiate PyAudio
audio = pyaudio.PyAudio()
#open stream

# f = wave.open(r"./test.wav", "rb")
f = wave.open(r"./data/spoken_numbers_pcm/7_Vicki_160.wav", "rb")

# out_stream = audio.open(format = audio.get_format_from_width(f.getsampwidth()),
out_stream = audio.open( format=pyaudio.paInt8, channels = f.getnchannels(), rate=f.getframerate(), output= True)
# out_stream = audio.open( format=pyaudio.paFloat32, channels = f.getnchannels(), rate=f.getframerate(), output= True)

# data0 = f.readframes(chunk)
# data=numpy.fromstring(data0, dtype='uint8') #// set in stream ^^ !!
  # pygame.sndarray.make_sound(data).play()
  # stream.write(data*volume*volume)
  # data = np.reshape(data, (data.shape[0]/2, 2))# split stereo
  # p(data.shape)
  # x=numpy.float32(x)/255.0
  # numpy.seterrcall(cat)
  # numpy.seterrcall(lambda x: print("XX"+x))
  # numpy.seterr(divide='ignore', invalid='ignore')#// NO!
  # numpy.seterr(all='raise')# Does not help
  # data0 = f.readframes(chunk)
  # stream.write(data)



def gensin(frequency, duration, sampRate):
    """ Frequency in Hz, duration in seconds and sampleRate in Hz"""
    cycles = np.linspace(0,duration*2*np.pi,num=duration*sampRate)
    wave = np.sin(cycles*frequency,dtype='float32')
    wave = wave*1000
    wave = wave.astype('int16')
    # wave = np.sin(cycles*frequency,dtype='int16')
    # t = np.divide(cycles,2*np.pi)

    return wave

volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 4   # in seconds, may be float
f = 441.0        # sine frequency, Hz, may be float
# generate samples, note conversion to float32 array
x0 = 0

out_stream.start_stream()
wav=0
while 1:
  x, y = mouse.position()
  if x!=x0:
    # samples = (np.sin(2*np.pi*np.arange(fs*duration)*x/fs)).astype(np.float32)
    samples = gensin(x,duration,fs)
    x0=x
    if wav: wav.stop()
    wav=pygame.sndarray.make_sound(samples)
    # stream.write(volume * samples)
    # t, samples = gensin(y, duration, fs)
    # stream.write(volume * samples)
    # wav.play()
    # wav.play(loops=100, maxtime=duration, fade_ms=10)
    wav.play(loops=100, fade_ms=30)
    print(x,y)

# data=np.array(range(chunk))
while 1:
  for i in range(len(data)):#range(chunk):
    data[i] = math.sin(0.1* i / x)
    if i > 40900: freq1 = int(freq1 * (len(data) - i) / 100.0) # avoid 'Popping'
  stream.write(volume*data)

#stop stream
stream.stop_stream()
stream.close()

#close PyAudio
# p.terminate()

while 1:
  pass
exit()


import sys
import tensorflow as tf
sess = tf.InteractiveSession()
import speech_data
speech = speech_data.read_data_sets("/data/speech/", one_hot=True)



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
  tf.initialize_all_variables().run()
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

