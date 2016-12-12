import os

import librosa #mfcc 1.st lib
# from scikits.talkbox.features import mfcc # 2'nd lib
import os.path
import python_speech_features
import numpy as np
import subprocess

AUTOMATIC_ALL_VOICES=False
# list voices: `say -v ?`
# good_voices = [] #AUTOMATIC !!
good_voices = """
Agnes
Alex
Allison
Ava
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
Susan
Tessa
Tom
Veena
Vicki
Victoria
""".split()

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
max_word_length = 20

def pad(vec, pad_to=max_word_length,one_hot=False):
	for i in range(0, pad_to - len(vec)):
		if one_hot: vec.append([terminal_symbol] * num_characters)
		else: vec.append(terminal_symbol)
	return vec


def char_to_class(c):
	# type: (char) -> int
	classe=(ord(c) - offset) % num_characters
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
	# z = list(z)  # py3
	z = pad(z, pad_to)
	# z = np.array(z)
	return z # "abd"->[1,2,4,0,0,0,0,0] padded


def check_voices():
	voice_infos=subprocess.check_output(["say", "-v?"]).split("\n")[:-2]
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
	# mel_features, mspec, spec = mfcc(wave, fs=sample_rate, nceps=20)
	mel_features=python_speech_features.mfcc(signal, numcep=26, nfilt=26*2,samplerate=sample_rate) # 3rd lib
	# print len(mel_features)
	# print len(mel_features[0])
	# print("---")
	np.save(path + "/mfcc/%s_%s_%r.npy" % (word,voice,rate), mel_features)


def generate_phonemes(word,  path):
	chars= string_to_int_word( word )
	np.save(path + "/chars/%s.npy"%word, chars)
	pronounced=subprocess.check_output(["./word_to_phonemes.swift", word])
	# phonemes= pronounced_to_phoneme_class(pronounced)
	# np.save(path + "/phones/%s.npy"%word, phonemes)


def generate(words, path):
	# generate a bunch of files for each word (with many voices, nuances):
	# spoken wav/ogg
	# spectograph
	# mfcc Mel-frequency cepstrum
	# pronounced phonemes
	if not os.path.exists(path): os.mkdir(path)
	out=open(path + "/words.list", "wt")
	for word in words:
		print("generating %s"%word)
		out.write("%s\n"%word)
		generate_phonemes(word, path)
		rate=120
		for voice in good_voices:
			try:
				generate_mfcc(voice, word, rate, path)
			except:
				pass  # ignore after debug!


# generates
# number/chars/1.npy
# number/mfcc/1_Kathy_120.npy for each voice
def spoken_numbers():
	path = "number"
	generate(map(str,range(0,10)),path)


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
