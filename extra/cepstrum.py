# PowerSpectrum = abs(fft(SpeechFrame,1024)).^2;
# AutoCorrelation = ifft(PowerSpectrum,1024);
# Cepstrum = ifft(log(PowerSpectrum),1024);
import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt

sample_rate, X = scipy.io.wavfile.read("0_Kathy_160.wav")
print("sample_rate",sample_rate)
ceps, mspec, spec = mfcc(X,fs=sample_rate, nceps=10)
print("mspec ",mspec)
print("ceps ",ceps )
print("num_ceps ",len(ceps))
num_ceps = len(ceps)

np.save("0_Kathy_160.ceps", ceps) # cache results so that ML becomes fast

X = []
# ceps = np.load("0_Kathy_160.ceps")
# plt.matshow(ceps.transpose())
# plt.matshow(X)
# plt.matshow(mspec.transpose())
spec=spec.transpose()
spec=spec[len(spec)/2:]
a0=np.average(spec[0:len(spec)/4])
a1=np.average(spec[len(spec)/4:len(spec)/2])
a0=np.amax(spec[0:len(spec)/4],axis=0)
a1=np.amax(spec[len(spec)/4:len(spec)/2],axis=0)
a2=np.amin(spec[0:len(spec)/4],axis=0)
a3=np.amin(spec[len(spec)/4:len(spec)/2],axis=0)
a4=np.argmax(spec[0:len(spec)/2])*5
# spec=spec[len(spec)/2:]
spec[0]=a0
spec[1]=a1
spec[2]=a2
spec[3]=a3
spec[4]=a4
plt.matshow(spec)
plt.show()
X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
Vx = np.array(X)
print(Vx)
# use Vx as input values vector for neural net, k-means, etc


# OTHER: https://github.com/jameslyons/python_speech_features .
#     features.mfcc() - Mel Frequency Cepstral Coefficients
#     features.fbank() - Filterbank Energies
#     features.logfbank() - Log Filterbank Energies
#     features.ssc() - Spectral Subband Centroids
