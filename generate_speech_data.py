#!/usr/bin/env python
import os

import librosa #mfcc 1.st lib
from scikits.talkbox.features import mfcc # 2'nd lib
# import python_speech_features # 3rd lib
import os.path
import numpy as np
import subprocess

AUTOMATIC_ALL_VOICES=False
# list voices: `say -v ?`
# good_voices = [] #AUTOMATIC !!
good_voices = """
Agnes
Alex
Allison
Daniel
Fred
Junior
Karen
Kate
Kathy
Lee
Moira
Oliver
Princess
Ralph
Samantha
Serena
Tessa
Tom
Veena
Vicki
Victoria
""".split()
# Ava Susan not found.


bad_voices = """
Albert
Bad\ News
Bahh
Bells
Boing
Bruce
Bubbles
Cellos
Deranged
Good\ News
Hysterical
Pipe\ Organ
Trinoids
Whisper
Zarvox
""".split()

low_quality = """
Agnes
Fred
Kathy
Princess
Ralph
""".split()

# Whisper + Albert Deranged Trinoids  >> later !
no_rate = ["Bad News", "Bells", "Good News", "Cellos", "Pipe\ Organ"]

num_characters = 32
terminal_symbol = 0
offset = 64  # starting with UPPER case characters ord('A'):65->1
max_word_length = 78# 20

def pad(vec, pad_to=max_word_length,one_hot=False):
	for i in range(0, pad_to - len(vec)):
		if one_hot: vec.append([terminal_symbol] * num_characters)
		else: vec.append(terminal_symbol)
	return vec


def char_to_class(c):
	# type: (char|int) -> int
	if not isinstance(c,int): c=ord(c)
	classe=(c - offset) % num_characters
	if c==' ' or c=="_": classe= terminal_symbol  # needed by ctc
	return classe # A->1 ... Z->26


# def achar_to_phoneme(c):
def pronounced_to_phoneme_class(pronounced):
	# type: (str) -> [int]
	raise Exception("TODO")
	phonemes = map(char_to_class, pronounced)
	z = pad(z, pad_to)
	return phonemes  # "_hEllO"->[11,42,24,21,0,0,0,0] padded


def string_to_int_word(word, pad_to=max_word_length):
	# type: (str) -> [int]
	z = map(char_to_class, word)
	z = list(z)  # py3
	z = pad(z, pad_to)
	# z = np.array(z)
	return z # "abd"->[1,2,4,0,0,0,0,0] padded


def check_voices():
	voice_infos=str(subprocess.check_output(["say", "-v?"])).split("\n")[:-2]
	voices=map(lambda x:x.split()[0],voice_infos)
	for voice in good_voices:
		if voice in voices:
			print (voice + " FOUND!")
	for voice in good_voices:
		if not voice in voices:
			print (voice+" MISSING!")
			good_voices.remove(voice)

	# ADD ALL ACCENTS!! YAY!!
	if AUTOMATIC_ALL_VOICES: # takes looong to create and is harder to train (really?)
		#  todo: add trainig difficulty metadata to samples!
		for voice in voices:
			good_voices.append(voice)


check_voices()

def generate_mfcc(voice, word, rate, path):
	filename = path+"/ogg/{0}_{1}_{2}.ogg".format(word, voice, rate)
	cmd = "say '{0}' -v{1} -r{2}  -o '{3}'".format(word, voice, rate, filename)
	os.system(cmd)  # ogg aiff m4a or caff
	signal, sample_rate = librosa.load(filename, mono=True)
	# mel_features = librosa.feature.mfcc(signal, sample_rate)
	# sample_rate, wave = scipy.io.wavfile.read(filename) # 2nd lib
	mel_features, mspec, spec = mfcc(signal, fs=sample_rate, nceps=26)
	# mel_features=python_speech_features.mfcc(signal, numcep=26, nfilt=26*2,samplerate=sample_rate) # 3rd lib
	# print len(mel_features)
	# print len(mel_features[0])
	# print("---")
	mel_features=np.swapaxes(mel_features,0,1)# timesteps x nFeatures -> nFeatures x timesteps
	np.save(path + "/mfcc/%s_%s_%d.npy" % (word,voice,rate), mel_features)

def generate_chars(voice, word, rate, path):
	chars = string_to_int_word(word) # todo : softlink!
	# os.symlink("%d.npy","%s_%s_%d.npy" % (word, voice, rate))
	np.save(path + "/chars/%s_%s_%d.npy" % (word, voice, rate), chars)

def generate_phonemes(word,  path):
	pronounced=subprocess.check_output(["./word_to_phonemes.swift", word]).decode('UTF-8').strip()
	chars = string_to_int_word(pronounced, pad_to=max_word_length)  # hack for numbers!
	# chars = string_to_int_word(word, pad_to=max_word_length)
	np.save(path + "/chars/%s.npy"%word, chars)
	# phonemes= pronounced_to_phoneme_class(pronounced)
	# np.save(path + "/phones/%s.npy"%word, phonemes)


def generate(words, path):
	# generate a bunch of files for each word (with many voices, nuances):
	# spoken wav/ogg
	# spectograph
	# mfcc Mel-frequency cepstrum
	# pronounced phonemes
	if not os.path.exists(path): os.mkdir(path)
	if not os.path.exists(path + "/chars/"): os.mkdir(path + "/chars/")
	if not os.path.exists(path + "/mfcc/"): os.mkdir(path + "/mfcc/")
	if not os.path.exists(path + "/ogg/"): os.mkdir(path + "/ogg/")
	out=open(path + "/words.list", "wt")
	for word in words:
		if isinstance(word, bytes):
			word=word.decode('UTF-8').strip()
		print("generating %s"%word)
		out.write("%s\n"%word)
		generate_phonemes(word, path)
		rate=120
		# for rate in range(80,360,step=20):
		for voice in good_voices:
			try:
				generate_chars(voice, word, rate, path)
				generate_mfcc(voice, word, rate, path)
			except:
				pass  # ignore after debug!


# generates
# number/chars/1.npy
# number/mfcc/1_Kathy_120.npy for each voice
def spoken_numbers():
	path = "number"
	nums = list(map(str, range(0, 10)))
	generate(nums, path)


def spoken_words():
	path = "words"
	wordlist = "wordlist.txt"
	words= open(wordlist).readlines()
	generate(words, path)


def spoken_sentence():
	path = "sentences"
	wordlist = "sentences.txt"
	words = open(wordlist).readlines()
	generate(words, path)


def extra():
	for v in bad_voices:
		for w in range(0,10):
			cmd = "say '{w}' -v'{v}' -r120"  # -o 'spoken_numbers/{w}_{v}.ogg'"
			os.system(cmd)

def main():
	spoken_numbers()
	# spoken_words()
	# spoken_sentence()

if __name__ == '__main__':
	main()
	print("DONE!")
