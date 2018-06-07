#!/usr/bin/env python
import subprocess
import skimage.io
import traceback
import numpy
import numpy as np
import os
import sys
from os import system
from platform import system as platform
import skimage.io
import wave
import pyaudio
import matplotlib.pyplot as plt


plt.matshow([[1,0],[0,1]], fignum=1)
plt.draw()

if platform() == 'Darwin':  # How Mac OS X is identified by Python
    system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

i = 0
width=256
height=256

# Number of bytes to be captured from audio stream
# CHUNK = 512
# CHUNK = 1024
# CHUNK = 1024
# CHUNK = 2048
CHUNK = 4096
# CHUNK = 9192

# number of bytes used per FFT fourier slice
# length=512
length = 1024
# length=2048
# length = 4096

#  forward step in sliding window [ CHUNK    [[length]-> ]step      CHUNK   ]
# step=32
# step=64
# step = 128
step=256
# step=512
# step<length : some overlap

image=numpy.array(bytearray(os.urandom(width*width)))
image=image.reshape(width,width)

def get_audio_input_stream():
  INDEX = 0  # 1
  # FORMAT = pyaudio.paInt8
  FORMAT = pyaudio.paInt16
  # FORMAT = pyaudio.paInt32
  # FORMAT = pyaudio.paFloat32
  CHANNELS = 1
  # RATE = 22500
  RATE = 48000 #* 2 = 96000Hz max on mac
  INPUT_BLOCK_TIME = 0.05
  # INPUT_BLOCK_TIME = 0.1
  INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)

  stream = pyaudio.PyAudio().open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    input_device_index=INDEX)
  return stream

  

def next_frame():
  stream = get_audio_input_stream()
  while True:
    try:
      dataraw = stream.read(CHUNK)
    except IOError as e:
      print(e) # [Errno -9981] Input overflowed  WHY?
      stream = get_audio_input_stream() # reset
      continue
    data0 = numpy.fromstring(dataraw, dtype='int16')
    yield data0

def record():
  global i
  global image
  global winName
  FILENAME = 'recording.wav'
  # r = numpy.array()
  hamming_window = np.hamming(length) # minimize fourier frequency drain
  #hamming hanning bartlett 'blackman'
  r = numpy.empty(length)
  stream = get_audio_input_stream()
  offset = 0
  while True:
    try:
	    dataraw = stream.read(CHUNK)
    except IOError as e:
	    print(e) # [Errno -9981] Input overflowed  WHY?
	    stream=get_audio_input_stream()
	    pass
    data0 = numpy.fromstring(dataraw, dtype='int16')
    # data0 = numpy.fromstring(dataraw, dtype='int8')
    if(i<20 and numpy.sum(np.abs(data0))<1000*width):
      continue
    r=numpy.append(r,data0)
    while offset < r.size - length :
      data = r[offset:offset+length]
      data=data*hamming_window  # minimize fourier frequency drain
      offset=offset + step

      data = numpy.fft.fft(data)#.abs()
      data = numpy.absolute(data)
      data = data[0:height]/256.0#.split(data,512)
      data = numpy.log2(data*0.05+1.0)#//*50.0;
      numpy.putmask(data, data > 255, 255)

      image[i] = data
      i = i+1
      if(i==width):
        print("i %d\r"%i)
        i=0
        # image=image.T
        image=numpy.rot90(image)
        plt.matshow(image, fignum=1)
        plt.draw()
        plt.pause(0.01)
        # result=spec2word(image) #todo: reconnect
        # subprocess.call(["say"," %s"%result])
        # cv2.imshow(winName,image)
        # if cv2.waitKey(10) == 27: BREAKS portAudio !!

if __name__ == '__main__':
  record()
