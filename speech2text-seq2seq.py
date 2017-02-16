#!/usr/bin/env python
#!/usr/local/bin/python


print("""
Update:
tf.nn.seq2seq doesn't work as hoped:
	It needs a 1D Tensor (chars) as input, not 2D spectrogram/mfcc/...
 unless we feed it with very long 1D wave data,
 but that is probably not what seq2seq was intended to for.
 Fear not: 1D dilated convolution and LSTMs together with CTC are just fine.

 New dynamic seq2seq was recently added to the master and will probably be out with 1.0.0 release.I
""")


exit(0)

"""Sequence-to-sequence model with an attention mechanism."""
from __future__ import print_function
import numpy as np
import tensorflow as tf
# import sugartensor as tf
# import sugartensor
import layer
import speech_data
from speech_data import Source,Target
from layer import net

learning_rate = 0.00001
training_iters = 300000 #steps
batch_size = 64



input_classes=20 # mfcc features
max_input_length=80 # (max) length of utterance
max_output_length=20
output_classes=32 # dimensions: characters


# Target.word here just returns the filename "1_STEFFI_160.wav" = digit_speaker_words-per-minute.wav nicely 'encoded' ;)
batch=word_batch=speech_data.mfcc_batch_generator(batch_size, source=Source.DIGIT_WAVES, target=Target.hotword)
X,Y=next(batch)

# EOS='\n' # end of sequence symbol todo use how?
# GO=1		 # start symbol 0x01 todo use how?
# def decode(bytes):
# 	return "".join(map(chr, bytes)).replace('\x00', '').replace('\n', '')

vocab_size=input_classes
target_vocab_size=output_classes
buckets=[(max_input_length, max_output_length)] # our input and response words can be up to 10 characters long
# (1000,1000) Takes 6 minutes on the Mac, half on Nvidia
PAD=[0] # fill words shorter than 10 characters with 'padding' zeroes

input_data    = x= X
target_data   = y= Y
target_weights= [[1.0]*50 + [0.0]*(max_input_length-50)] *batch_size # mask padding. todo: redundant --
encoder_size = max_input_length
decoder_size = max_output_length #self.buckets[bucket_id]

num_dim=input_classes #?

# residual block
def res_block(tensor, size, rate, dim=num_dim):
    # filter convolution
    conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True)
    # gate convolution
    conv_gate = tensor.sg_aconv1d(size=size, rate=rate,  act='sigmoid', bn=True)
    # output by gate multiplying
    out = conv_filter * conv_gate
    # final output
    out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True)
    # residual and skip output
    return out + tensor, out

# expand dimension
z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True)

# dilated conv block loop
skip = 0  # skip connections
for i in range(num_blocks):
    for r in [1, 2, 4, 8, 16]:
        z, s = res_block(z, size=7, rate=r)
        skip += s

# final logit layers
logit = (skip
         .sg_conv1d(size=1, act='tanh', bn=True)
         .sg_conv1d(size=1, dim=voca_size))

# CTC loss
loss = logit.sg_ctc(target=y, seq_len=seq_len)
tf.train.AdamOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver(tf.global_variables())

# train
tf.sg_train(log_interval=30, lr=0.0001, loss=loss, ep_size=1000, max_ep=200, early_stop=False)



# tf.nn.seq2seq DOES'T WORK: NEEDS 1D Tensor (chars) as input, not mfcc
# class SpeechSeq2Seq(object):
#
# 	def __init__(self,size, num_layers):
#
# 		cell = single_cell = tf.nn.rnn_cell.GRUCell(size)
# 		if num_layers > 1:
# 		 cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
# 		# Feeds for inputs.
# 		self.encoder_inputs = []
# 		self.decoder_inputs = []
# 		self.target_weights = []
# 		i=0
# 		self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
# 		self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
# 		self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
# 		# tf.nn.rnn()
# 		# targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]
# 		self.outputs, self.losses = tf.nn.seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)
# 		tf.train.AdamOptimizer(learning_rate).minimize(self.losses)
# 		self.saver = tf.train.Saver(tf.all_variables())
#
# 	# def step(self, session, encoder_inputs, decoder_inputs, target_weights, test):
# 		pass
# def test():
# 	perplexity, outputs = model.step(session, input_data, target_data, target_weights, test=True)
# 	words = np.argmax(outputs, axis=2)  # shape (10, 10, 256)
# 	# word = decode(words[0])
# 	word = str(words[0])
# 	print("step %d, perplexity %f, output: hello %s?" % (step, perplexity, word))

# def train():
# 	step=0
# 	test_step=1
# 	with tf.Session() as session:
# 		model= SpeechSeq2Seq(size=10, num_layers=1)
# 		session.run(tf.initialize_all_variables())
# 		while True:
# 			model.step(session, input_data, target_data, target_weights, test=False) # no outputs in training
# 			if step % test_step == 0:
# 				test()
# 			step=step+1
