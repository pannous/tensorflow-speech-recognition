#!/usr/bin/env python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import tflearn
import pyaudio
import speech_data
import numpy

# Simple spoken digit recognition demo, with 98% accuracy in under a minute

# Training Step: 544  | total loss: 0.15866
# | Adam | epoch: 034 | loss: 0.15866 - acc: 0.9818 -- iter: 0000/1000

batch=speech_data.wave_batch_generator(10000,target=speech_data.Target.digits)
X,Y=next(batch)

number_classes=10 # Digits

# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 8192])
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(X, Y,n_epoch=3,show_metric=True,snapshot_step=100)
# Overfitting okay for now

demo_file = "5_Vicki_260.wav"
demo=speech_data.load_wav_file(speech_data.path + demo_file)
result=model.predict([demo])
result=numpy.argmax(result)
print("predicted digit for %s : result = %d "%(demo_file,result))
