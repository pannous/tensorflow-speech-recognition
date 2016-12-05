import numpy as np
import tensorflow as tf
import layer
import speech_data
from speech_data import Source,Target
from layer import net

learning_rate = 0.00001
training_iters = 300000 #steps
batch_size = 64


width=20 # mfcc features
height=80 # (max) length of utterance
classes=10 # digits


batch=word_batch=speech_data.mfcc_batch_generator(batch_size, source=Source.DIGIT_WAVES, target=Target.digits)
X,Y=next(batch)

encoder_inputs=X
decoder_inputs=Y
# inputs = [tf.placeholder(tf.float32, shape=(batch_size, input_size)) for _ in xrange(10)]

cell=tf.nn.seq2seq.rnn_cell.LSTMCell(num_units=100,state_is_tuple=True,use_peepholes=True)

inputs=X
#  rnn
state = cell.zero_state(width,tf.float64) # not batch_size
outputs = []
states = []
i=0
for input_ in inputs:
	with tf.variable_scope('block',reuse=i>0):
		output, state = cell(tf.convert_to_tensor( input_), state)
		outputs.append(output)
		states.append(state)
	i=i+1
	# return (outputs, state)

# outputs, states = tf.nn.seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)
y_true=decoder_inputs
y_pred=outputs
# cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
cost = tf.reduce_mean(tf.pow(np.array(y_true) - np.array(y_pred), 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)
training_epochs=10
for epoch in range(training_epochs):
	_, c=sess.run(optimizer,cost)
	# ok=sess.run(outputs)
	# print(ok)
	print(c)
