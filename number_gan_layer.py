#!/usr/bin/env python
#!/usr/bin/python
import speech_data
import layer
from layer import *

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

batch=speech_data.wave_batch_generator(1000)
X,Y=next(batch)

# Classification
# x = tflearn.input_data(shape=[None, 8192])
# net = tflearn.fully_connected(x, 64)
# net = tflearn.dropout(net, 0.5)
# net = tflearn.fully_connected(net, 10, activation='softmax')
# net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
# y = net.placeholder
# classifier = tflearn.DNN(net)

def model(net):
	# type: (layer.net) -> None
	# net.inputnet_data(shape=[None, 10])
	net.fully_connected( 64)
	net.dropout( 0.5)
	net.fully_connected( 8192, activation='tanh') #sigmoid
	# net.regression(gan, optimizer='adam', loss='mean_square')#,placeholder=__)
# categorical_crossentropy, binary_crossentropy, softmax_categorical_crossentropy, hinge_loss, mean_square

# print("batch",batch)
net=net(model,data=batch)
net.train(display_step=1,test_step=10)
	# model.fit(X, Y,n_epoch=10,show_metric=True,snapshot_step=1000)
	# net.fit(Y,X,n_epoch=100, show_metric=True, snapshot_step=1000)
	# gananlyzer.fit(Y, n_epoch=10, show_metric=True, snapshot_step=1000)


