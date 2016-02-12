#!/usr/bin/env python
import cv
import numpy
import json            
import cv2
import os
import sys
import json
import re
import threading
import traceback   
import urllib2
import pyaudio
import skimage.io
from os import system
from platform import system as platform
# from json import dumps, loads, JSONEncoder, JSONDecoder

i = 0
xFactor=1
abort = False
lock = False
# image=numpy.array(bytearray(os.urandom(512*xFactor*512))) # 512,512)
image = numpy.zeros(512*xFactor*512).astype(numpy.uint8)
image = image.reshape(512*xFactor,512)    
image[0] = numpy.zeros(512)
image[1] = numpy.zeros(512)
image[2] = numpy.zeros(512)
last = image

winName="Record speech"
cv2.namedWindow(winName, cv.CV_WINDOW_FULLSCREEN)            

if platform() == 'Darwin':  # How Mac OS X is identified by Python
    system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

class RecordThread(threading.Thread):
         def run(self):
            record()

def say(what):
  os.system("say  %s"%what)          

def teach(lock):
  # blocking
  print "TEACHING %s",lock
  if(i>30):
    upload(image.T,lock)
  else:
    upload(last.T,lock)

def record():
  global i
  global image
  global winName
  global abort
  # INDEX = 1
  INDEX = 0
  FORMAT = pyaudio.paInt16
  # FORMAT = pyaudio.paInt8
  CHANNELS = 1
  # RATE = 48000
  # RATE = 44100
  RATE = 22050#Hz 1ch s16le LIKE say cmd!
    # Its the audio interface telling SoX it doesn't support that rate.  Its a very quirky interface.
    # OSX gives a loud warning during compile that the audio interface we are using has been deprecated for long time now and not to use it. 
    # CD sample rates are at 44100

  
  # RATE = 22500
  # INPUT_BLOCK_TIME = 0.05
  INPUT_BLOCK_TIME = 0.01
  INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)
  # CHUNK = 512
  # CHUNK = 1024
  # CHUNK = 1024
  # CHUNK = 2048
  CHUNK = 4096
  # CHUNK = 9192
  # length=512
  length=1024
  # length=2048
  # length = 4096
  # step=32
  step=64
  # step=128
  # step=256
  # print pyaudio.get_device_info_by_index(INDEX)
  
  zeroz=numpy.zeros(512)
  stream = pyaudio.PyAudio().open(
      format = FORMAT,
      channels = CHANNELS,
      rate = RATE,
      input = True,
      frames_per_buffer = CHUNK,
      input_device_index = INDEX )

  # r = numpy.array()
  offset = 0
  summ=0
  r = numpy.empty(length)
  while True:
    if abort:
      return
    try:
      dataraw = stream.read(CHUNK)
      data0 = numpy.fromstring(dataraw, dtype='int16')
      # data0 = numpy.fromstring(dataraw, dtype='int8')
      last=summ
      summ=numpy.sum(numpy.absolute(data0))
      if(lock):
        teach(lock)
      # print summ  

      if(i<20 and summ<23456): #  coarse filtering, good for anything?
        continue
      # if(i<20 and (summ<180 or last<180)):
      #        continue      
      # print summ
      # print 'go!'
      r=numpy.append(r,data0)
      while offset < r.size - length :
        if abort:
          return
        if(lock):
          teach(lock)
        data = r[offset:offset+length]
        offset=offset + step
        data = numpy.fft.fft(data)
        data = numpy.absolute(data)
        # data = data[0:512]/256.0 #/4 #WHY 4 ?? 2^16=2^8*...
        data = data[-512-1:-1]/256.0 #/4 #WHY 4 ?? 2^16=2^8*...
        summ=numpy.sum(data)
        if(summ<1000): 
          if i<30 :
            i=3#reset()
            # break
            continue
          else:
            while i<512:
              image[i] = zeroz
              i = i+1
        else:              
         numpy.putmask(data, data > 255, 255)
         
         # data = numpy.log2(data/(2^4)+1.0)*50.0;
         
        i = i+1
        if(i>=512*xFactor):
          threading.Thread(target=upload, args=[image.T]).start();
          cv2.imshow(winName,image.T)
          i=3
        else:
          image[i] = data 
          if(i%4==0):
            cv2.imshow(winName,image.T) 
          # result=upload(image)
          # print "YAY %s"%result
          # result=re.compile("(\\d)").search(result).group(1)
          # threading.Thread(target=say, args=[result]).start();
          
          # cv2.imwrite('snapshot/RandomGray%d.png'%i,image)
        # if cv2.waitKey(10) == 27: BREAKS portAudio !!
              # cv2.destroyWindow(winName)
              # return 0
              
    except IOError:
      print 'lost frame' # reduce imshow  frequency
    #   print 'todo: in threading'          
    except  Exception as err:
          print('Record sound error: %s' % err)
          traceback.print_exc(file=sys.stdout)

def upload(image=None,clazz=None):
    global lock
    lock=None # clear now!
    if image==None:
      image_file="/me/ai/phonemes/5_Karen_260.wav.spec.png"
    # image_file="/me/ai/phonemes/spoken_numbers/7_Karen_260.wav.spec.png"
      image = skimage.io.imread(image_file).astype(numpy.uint8) #float32 BOTH OK!
  
    post_data=json.dumps({'json':image.tolist(),'class':clazz,'net':'speech'})
    req = urllib2.Request('http://192.168.1.24:5000/classify_image', post_data)
    print "sent"
    response = urllib2.urlopen(req)
    result = response.read()
    result= result[1:-1]
    result = numpy.fromstring(result, dtype=float, count=-1, sep=' ')#.round()# astype(numpy.uint8)
    print "YAY %s"%result#.join("")
    return result

import time
import sys
if __name__ == '__main__':
  global abort,lock #= False
  cv2.imshow(winName,image )
  # threading.Thread(target=say, args=["hi"]).start();
  threading.Thread(target=os.system, args=["say 5"]).start();
  r=RecordThread()
  r.start()
  # print int('8') # 8
  # print 56==('8') # false :(
  # record()
  # upload()
  # transform_all()
  while True:
    key=cv2.waitKey(1)
    if key<0:
      continue
    if chr(key)=='q' or key==ord('q') or key==27: 
      print 'DONE'
      abort=True
      # r._Thread__stop()
      time.sleep(.32)
      cv2.destroyWindow(winName)
      break
    else:
      lock=chr(key)
      print "got locky %s"%lock
    key=-1
      # sys.exit(0)

  
  # app.run(debug=True, host='0.0.0.0', port=5000)
  # app.run(debug=False, host='0.0.0.0', port=5000)
