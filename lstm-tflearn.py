#!/usr/bin/env python
import tflearn
import speech_data
import tensorflow as tf

learning_rate = 0.0001
training_iters = 300000  # steps
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits

batch = word_batch = speech_data.mfcc_batch_generator(batch_size)

# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128*4, dropout=0.5)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0)

# Training

## add this "fix" for tensorflow version errors
for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES): tf.add_to_collection(tf.GraphKeys.VARIABLES, x )


while --training_iters > 0:
	X, Y = next(batch)
	trainX, trainY = X, Y
	testX, testY = X, Y  # overfit for now
	model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)

_y=model.predict(next(batch)[0])
model.save("tflearn.lstm.model")
print (_y)
