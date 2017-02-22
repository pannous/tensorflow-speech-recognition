#!/usr/local/bin/python
#!/usr/bin/env python
from __future__ import print_function

import time
import numpy
import pyaudio
from dateutil import parser
import wave, struct
audio = pyaudio.PyAudio()

import csv

# csvfile = open('qanda/qanda_2012_ep99_climate.csv')  # , newline='' is an invalid keyword
# spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
# for row in spamreader:
# 	try:
# 		time, ida, kind, speaker, text = row
# 		print(speaker)
# 	# print(', '.join(row))
# 	except:
# 		pass
import datetime

path="qanda_2012_ep99_climate.srt"
lines=open(path).readlines()
i=0
def milliseconds(offset):
	time=parser.parse(offset).time()
	return time.microsecond/1000 + 1000*(time.second + 60*(time.minute + 60* time.hour))


waveFile = wave.open('qanda_2012_ep99_climate.wav', 'r')
frameRate = waveFile.getframerate()
out_stream = audio.open(format=pyaudio.paInt16, channels=waveFile.getnchannels(), rate=frameRate, output=True)
out_stream.start_stream()

length = waveFile.getnframes()
print(frameRate)

CHUNK = 1024
def play(fro, to):
	if fro>to or (to-fro)>3*1000:
		print("segmented too long:")
		print(fro)
		print(to)
		return
	fro = fro* frameRate/1000
	to = to * frameRate /1000
	waveFile.setpos(fro)
	dataraw = waveFile.readframes(2 * (to - fro)) # 2 ?? channels?
	data0 = numpy.fromstring(dataraw, dtype='int16')
	out_stream.write(data0)
	# time.sleep(1)


# arr=[ for waveData in waveDatas]


while i<len(lines):
	nr,offsets,word=lines[i:i+3]
	print(word)
	fro, to = offsets.split("-->")
	fro = milliseconds(fro)
	to = milliseconds(to)
	play(fro,to)
	# print(fro)
	i += 4
	# break

