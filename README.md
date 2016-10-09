# Tensorflow Speech Recognition
Speech recognition using google's [tensorflow](https://github.com/tensorflow/tensorflow/) deep learning framework, [sequence-to-sequence](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html) neural networks.

Replaces [caffe-speech-recognition](https://github.com/pannous/caffe-speech-recognition), see there for some background.

## Ultimate goal
Create a decent standalone speech recognition for Linux etc.
Some people say we have the models but not enough training data.
We disagree: There is plenty of training data (100GB [here](http://www.openslr.org/12), on Gutenberg, synthetic Text to Speech snippets, Movies with transcripts, YouTube with captions etc etc) we just need a simple yet powerful model. It's only a question of time...


![Sample spectrogram, That's what she said, too laid?](https://raw.githubusercontent.com/pannous/caffe-speech-recognition/master/0_Karen_160.png)

Sample spectrogram, Karen uttering 'zero' with 160 words per minute.

## Getting started

Toy examples:
`./number_classifier_tflearn.py`
`./speaker_classifier_tflearn.py`

Some less trivial architectures:
`./densenet_layer.py`

Later:
`./train.sh`
`./record.py`


## Partners + Collaborators wanted
We are in the process of tackling this project in seriousness. Drop an email to info@pannous.com if you want to join the party, no matter your background.

Update: Nervana [demonstrated](https://www.youtube.com/watch?v=NaqZkV_fBIM) that it is possible for 'independents' to build models that are state of the art. Unfortunately they didn't open source the software.
[Sphinx starts using tensorflow LSTMs](http://cmusphinx.sourceforge.net/).


###Fun tasks for newcomers
* Data Augmentation :  create on-the-fly modulation of the data: increase the speech frequency, add background noise, alter the pitch etc,...


###Extensions 
**Extensions** to current tensorflow which are probably needed:
* Sliding window GPU implementation
* Continuous densenet->seq2seq adaptation
* Modular graphs/models + persistance
* Incremental collaborative snapshots ('P2P learning')
