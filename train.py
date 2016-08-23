"""
  training our speech model.
  
  The general architecture is:
  
  input=audio-wave-bytes
  preprocessed=phoneme-stream
  transcribed=word-stream
  
  ideally input->preprocessed will be just a subnet of a big model
  
  the intermediate step makes sense,  because we have data to train it separately. 
   it can even be subdivided into more specific curriculum learning steps:
   1) kind of FFT
   2) speech features / artifacts / internal representation vector
   3) phonemes
  
  The decode step will run STT backwards, thus (hopefully) yielding TTS
  
   As an intermediate step we can train an auto-encoder 
   from 0)input to 2)internal-representation and backwards
   
  The advantage of such an autoencoder is that we can train on arbitrary voice recordings without labels
  Ideally speech recognition (STT) would be helped by learning to speak (TTS) within the same model
   
"""
# from __future__ import solution
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import speech_data
import speech_model
from speech_encoder import train_spectrogram_encoder

tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,  "Batch size to use during training.")
# tf.app.flags.DEFINE_integer("phoneme_size", 200, "phoneme 'vocabulary' size.")
tf.app.flags.DEFINE_string("data_dir", "./data/", "Data directory")
tf.app.flags.DEFINE_integer("limit", 0, "Limit on the size of training data (0: no limit) / steps")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("task", "train_all", "Set to eval,tts,sst,train_...")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# phoneme_buckets = [(2, 4), (4, 6), (6, 8), (8, 12),(12, 20)]


def train_all():
  # while each of these methods can be implemented separately,
  # it is important that they operate on (subsets of) the same model!
  
  # also it is important that progress to the model will be uploaded,
  # so that the community constantly improves the model,
  # even if one individual can only contribute the processing power of one GPU/night
  while true:
    # train_autoencoder()  # wave -> wave
    train_spectrogram_encoder() # spectrogram -> vector
    # train_tts()   # text -> wave
    # train_internal_model() # wave -> vector
    # train_phonemes()    # phonemes <-> text (easy)
    # train_stt_words()   # wave snippet -> word
    # train_stt()         # wave stream -> text  , the whole package
    
def train_phonemes():
  """Train a phoneme->english translation model."""
  # Prepare WMT data.
  print("Preparing WMT data in %s" % FLAGS.data_dir)
  en_train, fr_train, _, _ = speech_data.prepare_data(
      FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading  training data (limit: %d)." % FLAGS.max_train_data_size)
    train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        
def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train_all()

if __name__ == "__main__":
  tf.app.run()
