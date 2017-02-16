#!/usr/bin/env python
import tflearn
import pyaudio
import speech_data


# Training Step: 544  | total loss: 0.15866
# | Adam | epoch: 034 | loss: 0.15866 - acc: 0.9818 -- iter: 0000/1000
# 98% Accuracy on training set in just a minute

# audio = pyaudio.PyAudio()
# # format=pyaudio.paFloat32
# format=pyaudio.paInt8
# # format=audio.get_format_from_width(f.getsampwidth())
# # out_stream = audio.open( format=format,channels = f.getnchannels(), rate=f.getframerate(), output= True)
# out_stream = audio.open( format=format,channels = 1, rate=48000, output= True)
# out_stream.start_stream()
# def play_pcm(data):
#   out_stream.write(data)

batch=speech_data.wave_batch(1000)
X,Y=next(batch)

# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)
#
# x = tflearn.input_data(shape=[None, 8192])
# net = tflearn.fully_connected(x, 64)
# net = tflearn.dropout(net, 0.5)
# net = tflearn.fully_connected(net, 10, activation='softmax')
# net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
# y = net.placeholder
# classifier = tflearn.DNN(net)

gan = tflearn.input_data(shape=[None, 10])
gan = tflearn.fully_connected(gan, 64)
gan = tflearn.dropout(gan, 0.5)
gan = tflearn.fully_connected(gan, 8192, activation='tanh') #sigmoid
gan = tflearn.regression(gan, optimizer='adam', loss='mean_square')#,placeholder=__)
# categorical_crossentropy, binary_crossentropy, softmax_categorical_crossentropy, hinge_loss, mean_square

gan = tflearn.DNN(gan)
while 1:
	# model.fit(X, Y,n_epoch=10,show_metric=True,snapshot_step=1000)
	gan.fit(Y,X,n_epoch=100, show_metric=True, snapshot_step=1000)
	# gananlyzer.fit(Y, n_epoch=10, show_metric=True, snapshot_step=1000)


