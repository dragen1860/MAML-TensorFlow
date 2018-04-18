""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
	if nb_samples is not None:
		sampler = lambda x: random.sample(x, nb_samples)
	else:
		sampler = lambda x: x
	images = [(i, os.path.join(path, image)) \
	          for i, path in zip(labels, paths) \
	          for image in sampler(os.listdir(path))]
	if shuffle:
		random.shuffle(images)
	return images


## Network helpers
def conv_block(x, weight, bias, reuse, scope):
	# conv
	x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME') + bias
	# batch norm
	x = tf_layers.batch_norm(x, activation_fn=tf.nn.relu, reuse=reuse, scope=scope)
	# pooling
	x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
	return x


def normalize(inp, activation, reuse, scope):
	if FLAGS.norm == 'batch_norm':
		return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
	elif FLAGS.norm == 'layer_norm':
		return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
	elif FLAGS.norm == 'None':
		if activation is not None:
			return activation(inp)
		else:
			return inp


## Loss functions
def mse(pred, label):
	pred = tf.reshape(pred, [-1])
	label = tf.reshape(label, [-1])
	return tf.reduce_mean(tf.square(pred - label))


def xent(pred, label):
	# Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
	return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size
