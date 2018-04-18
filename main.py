import csv, os
import numpy as np
import pickle
import random
import tensorflow as tf
from tensorflow import flags

from data_generator import DataGenerator
from maml import MAML

FLAGS = flags.FLAGS

flags.DEFINE_bool('train', True, 'train or test')
flags.DEFINE_integer('meta_iteration', 6000, 'iteraion for meta-train')
flags.DEFINE_integer('train_iteration', 5, 'iteraion for train update num')
flags.DEFINE_integer('meta_batchsz', 4, 'tasks num')
flags.DEFINE_integer('train_batchsz', 1, 'batchsz for one tasks, as we need test on same-domain train, here must be 1')
flags.DEFINE_float('meta_lr', 1e-3, 'meta-train learning rate, beta namely')
flags.DEFINE_float('train_lr', 1e-2, 'train learing rate, alpha namely')
flags.DEFINE_integer('nway', 5, 'n-way')
flags.DEFINE_integer('kshot', 1, 'k-shot')
flags.DEFINE_integer('kquery', 1, 'k-query, number of images to query per category')



def train(model, saver, sess):
	"""

	:param model:
	:param saver:
	:param sess:
	:return:
	"""

	train_writer = tf.summary.FileWriter(os.path.join('logs', 'mini.mdl'), sess.graph)
	prelosses, postlosses = [], []


	for iteration in range(FLAGS.meta_iteration):
		input_tensors = [model.meta_op]

		# add summary and print op
		if iteration % 100 == 0:
			input_tensors.extend([model.summ_op, model.support_loss, model.query_losses[FLAGS.train_iteration - 1]])
			input_tensors.extend([model.support_acc, model.query_accs[FLAGS.train_iteration - 1]])

		# run op
		result = sess.run(input_tensors, {})

		# summary
		if iteration % 100 == 0:
			prelosses.append(result[-2])
			train_writer.add_summary(result[1], iteration)
			postlosses.append(result[-1])

		# print
		if iteration % 100 == 0:
			print('loss pre&post:', np.mean(prelosses) , np.mean(postlosses))
			prelosses, postlosses = [], []

		# checkpoint
		if iteration % 100 == 0:
			saver.save(sess, os.path.join('logs', 'mini.mdl'))

		# evaluation
		if iteration % 200 == 220:
			result = sess.run([ model.metaval_total_accuracy1,
				                model.metaval_total_accuracies2[FLAGS.num_updates - 1],
				                model.summ_op],
			                  {})
			print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))




def main():

	if FLAGS.train:
		test_num_updates = 1  # eval on at least one update during training
	else:
		test_num_updates = 10 # test before 10 steps of fine-tuning
		orig_meta_batchsz = FLAGS.meta_batchsz
		# always use meta batch size of 1 when testing.
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
		support_x   = tf.slice(image_tensor, [0, 0, 0], [-1, FLAGS.nway * FLAGS.kshot, -1])
		query_x     = tf.slice(image_tensor, [0, FLAGS.nway * FLAGS.kshot, 0], [-1, -1, -1])
		support_y   = tf.slice(label_tensor, [0, 0, 0], [-1, FLAGS.nway * FLAGS.kshot, -1])
		query_y     = tf.slice(label_tensor, [0, FLAGS.nway * FLAGS.kshot, 0], [-1, -1, -1])
		input_train = {'support_x': support_x, 'support_y': support_y, 'query_x': query_x, 'query_y': query_y}

	# construct test tensors.
	random.seed(6)
	image_tensor, label_tensor = db.make_data_tensor(train=False)
	support_x   = tf.slice(image_tensor, [0, 0, 0], [-1, FLAGS.nway * FLAGS.kshot, -1])
	query_x     = tf.slice(image_tensor, [0, FLAGS.nway * FLAGS.kshot, 0], [-1, -1, -1])
	support_y   = tf.slice(label_tensor, [0, 0, 0], [-1, FLAGS.nway * FLAGS.kshot, -1])
	query_y     = tf.slice(label_tensor, [0, FLAGS.nway * FLAGS.kshot, 0], [-1, -1, -1])
	input_test = {'support_x': support_x, 'support_y': support_y, 'query_x': query_x, 'query_y': query_y}


	# dim_input: 84*84*3
	# dim_output: 5
	# 1. construct MAML model
	model = MAML(db.dim_input, db.dim_output, test_num_updates=test_num_updates)

	# construct metatrain_ and metaval_
	if FLAGS.train:
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


	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()

	# # load checkpoint from file
	# model_file = tf.train.latest_checkpoint(os.path.join('logs', 'mini.mdl'))
	# ind1 = model_file.index('model')
	# resume_itr = int(model_file[ind1 + 5:])
	# print("Restoring model weights from " + model_file)
	# saver.restore(sess, model_file)

	if FLAGS.train:
		train(model, saver, sess)




if __name__ == "__main__":
	main()
