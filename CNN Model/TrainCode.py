from __future__ import division

import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import resampy
from scipy.io import wavfile
import glob
import mel_features
from datetime import datetime
from random import shuffle


### Defining Directories
pos_dir = 'Dataset/balanced/gunshots/'
neg_dir = 'Dataset/balanced/negative/'
eval_pos_dir = 'Dataset/evaluation/gunshots/'
eval_neg_dir = 'Dataset/evaluation/negative/' 
checkpoint_dir = 'trained_checkpoint/'
result_dir = 'results/'


NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.
NUM_CLASSES = 2

SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.

LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.



def wavfile_to_examples(wav_file):
  
	sample_rate, wav_data = wavfile.read(wav_file)
	assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
	data = wav_data / 32768.0 # Convert to [-1.0, +1.0]

	# Convert to mono.
	if len(data.shape) > 1:
		data = np.mean(data, axis=1)
	# Resample to the rate assumed by VGGish.
	if sample_rate != SAMPLE_RATE:
		data = resampy.resample(data, sample_rate, SAMPLE_RATE)

	# Compute log mel spectrogram features.
	log_mel = mel_features.log_mel_spectrogram(data,
											audio_sample_rate= SAMPLE_RATE,
											log_offset= LOG_OFFSET,
											window_length_secs= STFT_WINDOW_LENGTH_SECONDS,
											hop_length_secs= STFT_HOP_LENGTH_SECONDS,
											num_mel_bins= NUM_MEL_BINS,
											lower_edge_hertz= MEL_MIN_HZ,
											upper_edge_hertz= MEL_MAX_HZ)

	# Frame features into examples.
	features_sample_rate = 1.0 /  STFT_HOP_LENGTH_SECONDS
	example_window_length = int(round( EXAMPLE_WINDOW_SECONDS * features_sample_rate))
	example_hop_length = int(round( EXAMPLE_HOP_SECONDS * features_sample_rate))
	log_mel_examples = mel_features.frame(log_mel,
										window_length=example_window_length,
	
										hop_length=example_hop_length)

	return log_mel_examples


def get_examples_batch(pos_data_file, neg_data_file):
  """Returns a shuffled batch of examples of all audio classes.
  Note that this is just a toy function because this is a simple demo intended
  to illustrate how the training code might work.
  Returns:
    a tuple (features, labels) where features is a NumPy array of shape
    [batch_size, num_frames, num_bands] where the batch_size is variable and
    each row is a log mel spectrogram patch of shape [num_frames, num_bands]
    suitable for feeding VGGish, while labels is a NumPy array of shape
    [batch_size, num_classes] where each row is a multi-hot label vector that
    provides the labels for corresponding rows in features.
  """
  # # Make a waveform for each class.
  # num_seconds = 5
  # sr = 44100  # Sampling rate.
  # t = np.linspace(0, num_seconds, int(num_seconds * sr))  # Time axis.
  # # Random sine wave.
  # freq = np.random.uniform(100, 1000)
  # sine = np.sin(2 * np.pi * freq * t)
  # # Random constant signal.
  # magnitude = np.random.uniform(-1, 1)
  # const = magnitude * t
  # # White noise.
  # noise = np.random.normal(-1, 1, size=t.shape)

  # Make examples of each signal and corresponding labels.
  # Sine is class index 0, Const class index 1, Noise class index 2.
  pos_examples = wavfile_to_examples(pos_data_file)
  pos_labels = np.array([[1, 0]] * pos_examples.shape[0])
  neg_examples = wavfile_to_examples(neg_data_file)
  neg_labels = np.array([[0, 1]] * neg_examples.shape[0])
  # noise_examples = vggish_input.waveform_to_examples(noise, sr)
  # noise_labels = np.array([[0, 0, 1]] * noise_examples.shape[0])

  # Shuffle (example, label) pairs across all classes.
  all_examples = np.concatenate((pos_examples, neg_examples))
  all_labels = np.concatenate((pos_labels, neg_labels))
  labeled_examples = list(zip(all_examples, all_labels))
  shuffle(labeled_examples)

  # Separate and return the features and labels.
  features = [example for (example, _) in labeled_examples]
  labels = [label for (_, label) in labeled_examples]
  return (features, labels)


# def get_dataset(pos_path, neg_path):

# 	for i in range(len(pos_path)):

# 		features,lables = get_examples_batch(pos_path[i], neg_path[i])








def network(input):

	# Input: a batch of 2-D log-mel-spectrogram patches.
    # features = tf.placeholder(tf.float32, shape=(None, NUM_FRAMES, NUM_BANDS), name='input_features')
    # Reshape to 4-D so that we can convolve a batch with conv2d().
    # net = tf.reshape(features, [-1, NUM_FRAMES, NUM_BANDS, 1])

    # The VGG stack of alternating convolutions and max-pools.
    net = slim.conv2d(input, 64, scope='conv1')
    net = slim.max_pool2d(net, scope='pool1')
    net = slim.conv2d(net, 128, scope='conv2')
    net = slim.max_pool2d(net, scope='pool2')
    net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3')
    net = slim.max_pool2d(net, scope='pool3')
    net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
    net = slim.max_pool2d(net, scope='pool4')
    net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
    net = slim.max_pool2d(net, scope='pool4')

    # Flatten before entering fully-connected layers
    net = slim.flatten(net)
    net = slim.repeat(net, 2, slim.fully_connected, 2, scope='fc1')

    # The embedding layer.
    net = slim.fully_connected(net, EMBEDDING_SIZE, scope='fc2')
    
    return tf.identity(net, name='embedding')

def vgg16(inputs):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.fully_connected(net, 4096, scope='fc6')
    net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc7')
    net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.fully_connected(net, 2, activation_fn=None, scope='fc8')

  return net


def define_vggish_slim(training=False):
  """Defines the VGGish TensorFlow model.
  All ops are created in the current default graph, under the scope 'vggish/'.
  The input is a placeholder named 'vggish/input_features' of type float32 and
  shape [batch_size, num_frames, num_bands] where batch_size is variable and
  num_frames and num_bands are constants, and [num_frames, num_bands] represents
  a log-mel-scale spectrogram patch covering num_bands frequency bands and
  num_frames time frames (where each frame step is usually 10ms). This is
  produced by computing the stabilized log(mel-spectrogram + params.LOG_OFFSET).
  The output is an op named 'vggish/embedding' which produces the activations of
  a 128-D embedding layer, which is usually the penultimate layer when used as
  part of a full model with a final classifier layer.
  Args:
    training: If true, all parameters are marked trainable.
  Returns:
    The op 'vggish/embeddings'.
  """
  # Defaults:
  # - All weights are initialized to N(0, INIT_STDDEV).
  # - All biases are initialized to 0.
  # - All activations are ReLU.
  # - All convolutions are 3x3 with stride 1 and SAME padding.
  # - All max-pools are 2x2 with stride 2 and SAME padding.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_initializer=tf.truncated_normal_initializer(
                          stddev= INIT_STDDEV),
                      biases_initializer=tf.zeros_initializer(),
                      activation_fn=tf.nn.relu,
                      trainable=training), \
       slim.arg_scope([slim.conv2d],
                      kernel_size=[3, 3], stride=1, padding='SAME'), \
       slim.arg_scope([slim.max_pool2d],
                      kernel_size=[2, 2], stride=2, padding='SAME'), \
       tf.variable_scope('vggish'):
    # Input: a batch of 2-D log-mel-spectrogram patches.
    features = tf.placeholder(
        tf.float32, shape=(None,  NUM_FRAMES,  NUM_BANDS),
        name='input_features')
    # Reshape to 4-D so that we can convolve a batch with conv2d().
    net = tf.reshape(features, [-1,  NUM_FRAMES,  NUM_BANDS, 1])

    # The VGG stack of alternating convolutions and max-pools.
    net = slim.conv2d(net, 64, scope='conv1')
    net = slim.max_pool2d(net, scope='pool1')
    net = slim.conv2d(net, 128, scope='conv2')
    net = slim.max_pool2d(net, scope='pool2')
    net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3')
    net = slim.max_pool2d(net, scope='pool3')
    net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
    net = slim.max_pool2d(net, scope='pool4')

    # Flatten before entering fully-connected layers
    net = slim.flatten(net)
    net = slim.repeat(net, 2, slim.fully_connected, 4096, scope='fc1')
    # The embedding layer.
    net = slim.fully_connected(net,  EMBEDDING_SIZE, scope='fc2')
    return tf.identity(net, name='embedding')


def main(_):

  with tf.Graph().as_default(), tf.Session() as sess:

  	# Defining session Intializer
	init = tf.global_variables_initializer()

	## Defining checkpoint file 
	saver = tf.train.Saver()

	# Saving checkpoint file for 
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt:
	    print('loaded ' + ckpt.model_checkpoint_path)
	    saver.restore(sess, ckpt.model_checkpoint_path)

	# Intializing the training iterations required
	train_iter = len(glob.glob(input_dir + '*.tif'))

	#Intializing last epoch to train with
	allfolders = glob.glob('./result/*0')
	lastepoch = 0
	for folder in allfolders:
		lastepoch = np.maximum(lastepoch, int(folder[-4:]))


    # Define VGGish.
    embeddings = vggish_slim.define_vggish_slim(training = True)

    # Define a shallow classification model and associated training ops on top
    # of VGGish.
    with tf.variable_scope('mymodel'):
      # Add a fully connected layer with 100 units.
      num_units = 100
      fc = slim.fully_connected(embeddings, num_units)

      # Add a classifier layer at the end, consisting of parallel logistic
      # classifiers, one per class. This allows for multi-class tasks.
      logits = slim.fully_connected(fc, _NUM_CLASSES, activation_fn=None, scope='logits')
      tf.sigmoid(logits, name='prediction')

      # Add training ops.
      with tf.variable_scope('train'):
        
        # global_step = tf.Variable(0, name='global_step', trainable=False,collections=[tf.GraphKeys.GLOBAL_VARIABLES,tf.GraphKeys.GLOBAL_STEP])

        # Labels are assumed to be fed as a batch multi-hot vectors, with
        # a 1 in the position of each positive class label, and 0 elsewhere.
        labels = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='labels')

        # Cross-entropy label loss.
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( logits=logits, labels=labels, name='xent'), name='loss_op')
        tf.summary.scalar('loss', loss)

        # We use the same optimizer and hyperparameters as used to train VGGish.
        optimizer = tf.train.AdamOptimizer(learning_rate= LEARNING_RATE, epsilon= ADAM_EPSILON)
        optimizer.minimize(loss, global_step=global_step, name='train_op')

    # Initialize all variables in the model, and then load the pre-trained
    # VGGish checkpoint.
    sess.run(tf.global_variables_initializer())
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

    # Locate all the tensors and ops we need for the training loop.
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
    # global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
    loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
    train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

    # The training loop.
    for index in range(lastepoch, 500):


      (features, labels) = get_examples_batch()
      [ loss, _] = sess.run([ loss_tensor, train_op],feed_dict={features_tensor: features, labels_tensor: labels})
      print('Step %d: loss %g' % (num_steps, loss))

if __name__ == '__main__':
  tf.app.run()

# with tf.Session() as sess:

# 	# Defining placeholders for features and associated labels.
# 	# features = tf.placeholder(tf.float32, shape=(None, NUM_FRAMES, NUM_BANDS), name='input_features')
# 	# features = tf.reshape(features, [-1, NUM_FRAMES, NUM_BANDS, 1])
# 	embeddings = define_vggish_slim(training = True)

# 	# Add a fully connected layer with 100 units.
# 	fc = slim.fully_connected(embeddings, num_units)

# 	# Labels are assumed to be fed as a batch multi-hot vectors, with a 1 in the position of each positive class label, and 0 elsewhere.
# 	labels = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='labels')


# 	# Add a classifier layer at the end, consisting of parallel logistic
# 	# classifiers, one per class. This allows for multi-class tasks.
# 	logits = slim.fully_connected(fc, NUM_CLASSES, activation_fn=None, scope='logits')
# 	tf.sigmoid(logits, name='prediction')

# 	# Defining loss/cost function and optimizer to use for training

# 	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), name='loss_op')
# 	optimizer = tf.train.AdamOptimizer( learning_rate= LEARNING_RATE, epsilon= ADAM_EPSILON).minimize(loss, name='train_op')

# 	## Defining session Intializer
# 	init = tf.global_variables_initializer()

# 	## Defining checkpoint file 
# 	saver = tf.train.Saver()




# 	# Intializing Tensorflow Session
# 	sess.run(init)

# 	# Saving checkpoint file for 
# 	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
# 	if ckpt:
# 	    print('loaded ' + ckpt.model_checkpoint_path)
# 	    saver.restore(sess, ckpt.model_checkpoint_path)

# 	#Intializing step
# 	step = 0

# 	# Intializing the training iterations required
# 	train_iter = len(glob.glob(input_dir + '*.tif'))

# 	#Intializing last epoch to train with
# 	allfolders = glob.glob('./result/*0')
# 	lastepoch = 0
# 	for folder in allfolders:
# 		lastepoch = np.maximum(lastepoch, int(folder[-4:]))


# 	for epoch in range(lastepoch, 500):

# 		#Creating a result directory to keep track of the current epoch running.
# 		os.path.isdir("result/%04d" % epoch)

# 		for index in range(train_iter):


# 		# Intializing the paths for positive and negative samples
# 		pos_path = (glob.glob(pos_dir + '*.wav'))
# 		neg_path = (glob.glob(neg_dir + '*.wav'))


# 		#Extracting the features and labels for each audio file
# 		features,lables = get_examples_batch(pos_path[index], neg_path[index])


# 		#Feeding the features and labels into the network
# 		[num_steps, loss, _] = sess.run([global_step_tensor, loss_tensor, train_op],feed_dict={features_tensor: features, labels_tensor: labels})


# pos_path = (glob.glob(pos_dir + '*.wav'))
# neg_path = (glob.glob(neg_dir + '*.wav'))
# # pos_feature, pos_labels = 
# features,lables = get_examples_batch(pos_path[0], neg_path[0])

# print(slim.fully_connected(define_vggish_slim(training=True), 100))

# # print(vgg16(features))
# # print(lables)

# print(len(features))
# print(len(lables))




