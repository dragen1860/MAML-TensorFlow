import os
import numpy as np
import pickle
import random
import tensorflow as tf
from tensorflow import flags

from data_generator import DataGenerator
from maml import MAML

FLAGS = flags.FLAGS

flags.DEFINE_bool('train', True, 'train or test')
flags.DEFINE_integer('meta_iteration', 600000, 'iteraion for meta-train')
flags.DEFINE_integer('train_iteration', 5, 'iteraion for train update num')
flags.DEFINE_integer('meta_batchsz', 4, 'tasks num')
flags.DEFINE_integer('train_batchsz', 1,
                     'should be 1. batchsz for one tasks, as we need test on same-domain train, here must be 1')
flags.DEFINE_float('meta_lr', 1e-3, 'meta-train learning rate, beta namely')
flags.DEFINE_float('train_lr', 1e-2, 'train learing rate, alpha namely')
flags.DEFINE_integer('nway', 5, 'n-way')
flags.DEFINE_integer('kshot', 1, 'k-shot')
flags.DEFINE_integer('kquery', 15, 'k-query, number of images to query per category')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train(model, saver, sess):
	"""

	:param model:
	:param saver:
	:param sess:
	:return:
	"""

	tb = tf.summary.FileWriter(os.path.join('logs', 'mini'), sess.graph)
	prelosses, postlosses = [], []

	# train for meta_iteartion epoches
	for iteration in range(FLAGS.meta_iteration):
		# this is the main op
		ops = [model.meta_op]

		# add summary and print op
		if iteration % 200 == 0:
			ops.extend([model.summ_op,
			            model.query_losses[0], model.query_losses[-1],
			            model.query_accs[0], model.query_accs[-1]])

		# run all ops
		result = sess.run(ops)

		# summary
		if iteration % 200 == 0:
			# query_acc[0]
			prelosses.append(result[-2])
			# summ_op
			tb.add_summary(result[1], iteration)
			# query_accs[-1]
			postlosses.append(result[-1])

			print('pre & post query loss:', iteration, np.mean(prelosses), np.mean(postlosses))
			prelosses, postlosses = [], []

		# checkpoint
		if iteration % 5000 == 0:
			saver.save(sess, os.path.join('ckpt', 'mini.mdl'))
			print('saved ckpt.')

		# evaluation
		if iteration % 1000 == 0:
			result = sess.run([model.test_query_accs[0],
			                   model.test_query_accs[-1]])
			print('Validation results: ', result[0], result[1])


def test(model, sess):

	np.random.seed(1)
	random.seed(1)

	# repeat test accuracy for 600 times
	test_accs = []
	feed_dict = {model.meta_lr: 0.0}
	for i in range(600):
		if i % 100 == 1:
			print(i)
		# extend return None!!!
		ops = [model.test_support_acc]
		ops.extend(model.test_query_accs)
		result = sess.run(ops, feed_dict=feed_dict)
		test_accs.append(result)

	# [600, K+1]
	test_accs = np.array(test_accs)
	# [K+1]
	means = np.mean(test_accs, 0)
	stds = np.std(test_accs, 0)
	ci95 = 1.96 * stds / np.sqrt(600)

	print('[support_t0, query_t0 - \t\t\ttK] ')
	print('mean:', means)
	print('stds:', stds)
	print('ci95:', ci95)



def main():
	if FLAGS.train:
		test_num_updates = 1  # eval on at least one update during training
	else:
		test_num_updates = 10  # test before 10 steps of fine-tuning
		orig_meta_batchsz = FLAGS.meta_batchsz
		# always use meta batch size of 1 when testing.
		# which means, we only build one task's graph
		FLAGS.meta_batchsz = 1

	# kshot + kquery images per category, nway categories, meta_batchsz tasks.
	db = DataGenerator(FLAGS.kshot + FLAGS.kquery, FLAGS.meta_batchsz)

	if FLAGS.train:  # only construct training model if needed
		random.seed(5)
		# get the tensor
		# image_tensor: [4, 80, 84*84*3]
		# label_tensor: [4, 80, 5]
		image_tensor, label_tensor = db.make_data_tensor()

		# NOTICE: the image order in 80 images should like this now:
		# [label2, label1, label3, label0, label4, and then repeat by 15 times, namely one task]
		# support_x : [4, 1*5, 84*84*3]
		# query_x   : [4, 15*5, 84*84*3]
		# support_y : [4, 5, 5]
		# query_y   : [4, 15*5, 5]
		support_x = tf.slice(image_tensor, [0, 0, 0], [-1, FLAGS.nway * FLAGS.kshot, -1])
		query_x = tf.slice(image_tensor, [0, FLAGS.nway * FLAGS.kshot, 0], [-1, -1, -1])
		support_y = tf.slice(label_tensor, [0, 0, 0], [-1, FLAGS.nway * FLAGS.kshot, -1])
		query_y = tf.slice(label_tensor, [0, FLAGS.nway * FLAGS.kshot, 0], [-1, -1, -1])
		input_train = {'support_x': support_x, 'support_y': support_y, 'query_x': query_x, 'query_y': query_y}

	# construct test tensors.
	random.seed(6)
	image_tensor, label_tensor = db.make_data_tensor(train=False)
	support_x = tf.slice(image_tensor, [0, 0, 0], [-1, FLAGS.nway * FLAGS.kshot, -1])
	query_x = tf.slice(image_tensor, [0, FLAGS.nway * FLAGS.kshot, 0], [-1, -1, -1])
	support_y = tf.slice(label_tensor, [0, 0, 0], [-1, FLAGS.nway * FLAGS.kshot, -1])
	query_y = tf.slice(label_tensor, [0, FLAGS.nway * FLAGS.kshot, 0], [-1, -1, -1])
	input_test = {'support_x': support_x, 'support_y': support_y, 'query_x': query_x, 'query_y': query_y}

	# dim_input: 84*84*3
	# dim_output: 5
	# 1. construct MAML model
	model = MAML(db.dim_input, db.dim_output, test_num_updates=test_num_updates)

	# construct metatrain_ and metaval_
	if FLAGS.train:
		# if not train, we only build test graph.
		model.build(input_train, prefix='metatrain_')
	model.build(input_test, prefix='metaval_')
	model.summ_op = tf.summary.merge_all()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)
	saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

	if not FLAGS.train:
		# change to original meta batch size when loading model.
		FLAGS.meta_batchsz = orig_meta_batchsz

	# initialize, under interative session
	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()

	# alway load ckpt both train and test.
	model_file = tf.train.latest_checkpoint('ckpt')
	print("Restoring model weights from ", model_file)
	saver.restore(sess, model_file)


	if FLAGS.train:
		train(model, saver, sess)
	else:
		test(model, sess)


if __name__ == "__main__":
	main()
