#!/usr/local/bin/python
from __future__ import print_function

import wave

import numpy
import pyaudio
from dateutil import parser

audio = pyaudio.PyAudio()

waveFile = wave.open('qanda_2012_ep99_climate.wav', 'r')
subtitles = "qanda_2012_ep99_climate.srt"

# csvfile = open('qanda/qanda_2012_ep99_climate.csv')  # , newline='' is an invalid keyword
# spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
# for row in spamreader:
# 	try:
# 		time, ida, kind, speaker, text = row
# 		print(speaker)
# 	except:
# 		pass

frameRate = waveFile.getframerate()
out_stream = audio.open(format=pyaudio.paInt16, channels=waveFile.getnchannels(), rate=frameRate, output=True)
out_stream.start_stream()

def play(fro, to):
	fro = fro* frameRate/1000
	to = to * frameRate /1000
	waveFile.setpos(fro)
	dataraw = waveFile.readframes(2 * (to - fro))  # 2 bytes per frame : 16bit little-endian mono @ 16khz
	data0 = numpy.fromstring(dataraw, dtype='int16')
	out_stream.write(data0)


# yield data0


def milliseconds(offset):
	time = parser.parse(offset).time()
	return time.microsecond / 1000 + 1000 * (time.second + 60 * (time.minute + 60 * time.hour))


lines = open(subtitles).readlines()
i = 0
while i<len(lines):
	nr,offsets,word=lines[i:i+3]
	i += 4
	print(word)
	fro, to = offsets.split("-->")
	fro = milliseconds(fro)
	to = milliseconds(to)
	if fro > to or (to - fro) > 3 * 1000:
		print("segment too long:")
		print(word)
		print(fro)
		print(to)
		continue
	play(fro,to)

