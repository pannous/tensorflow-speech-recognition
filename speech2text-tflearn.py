#!/usr/bin/env python
#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tflearn

import speech_data

import os

learning_rate = 0.0001
training_iters = 300000  # steps
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits

batch = word_batch = speech_data.mfcc_batch_generator(batch_size)
X, Y = next(batch)

# train, test, _ = ,X
trainX, trainY = X, Y
testX, testY = X, Y #overfit for now

# Data preprocessing
# Sequence padding
# trainX = pad_sequences(trainX, maxlen=100, value=0.)
# testX = pad_sequences(testX, maxlen=100, value=0.)
# # Converting labels to binary vectors
# trainY = to_categorical(trainY, nb_classes=2)
# testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, width, height])
# net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
# check model exists or not
if os.path.isfile("tflearn.lstm.model"):
    print ("loading model tflearn.lstm.model")
    model.load("tflearn.lstm.model")
while 1: #training_iters
    model.fit(trainX, trainY, n_epoch=100, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)
    _y=model.predict(X)
model.save("tflearn.lstm.model")
print (_y)
print (Y)

