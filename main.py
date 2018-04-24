import os
import numpy as np
import argparse
import random
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true', default=False, help='set for test, otherwise train')
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model, saver, sess):
	"""

	:param model:
	:param saver:
	:param sess:
	:return:
	"""
	# write graph to tensorboard
	# tb = tf.summary.FileWriter(os.path.join('logs', 'mini'), sess.graph)
	prelosses, postlosses, preaccs, postaccs = [], [], [], []
	best_acc = 0

	# train for meta_iteartion epoches
	for iteration in range(600000):
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
			# summ_op
			# tb.add_summary(result[1], iteration)
			# query_losses[0]
			prelosses.append(result[2])
			# query_losses[-1]
			postlosses.append(result[3])
			# query_accs[0]
			preaccs.append(result[4])
			# query_accs[-1]
			postaccs.append(result[5])

			print(iteration, '\tloss:', np.mean(prelosses), '=>', np.mean(postlosses),
			      '\t\tacc:', np.mean(preaccs), '=>', np.mean(postaccs))
			prelosses, postlosses, preaccs, postaccs = [], [], [], []

		# evaluation
		if iteration % 2000 == 0:
			# DO NOT write as a = b = [], in that case a=b
			# DO NOT use train variable as we have train func already.
			acc1s, acc2s = [], []
			# sample 20 times to get more accurate statistics.
			for _ in range(200):
				acc1, acc2 = sess.run([model.test_query_accs[0],
				                        model.test_query_accs[-1]])
				acc1s.append(acc1)
				acc2s.append(acc2)

			acc = np.mean(acc2s)
			print('>>>>\t\tValidation accs: ', np.mean(acc1s), acc, 'best:', best_acc, '\t\t<<<<')

			if acc - best_acc > 0.05 or acc > 0.4:
				saver.save(sess, os.path.join('ckpt', 'mini.mdl'))
				best_acc = acc
				print('saved into ckpt:', acc)


def test(model, sess):

	np.random.seed(1)
	random.seed(1)

	# repeat test accuracy for 600 times
	test_accs = []
	for i in range(600):
		if i % 100 == 1:
			print(i)
		# extend return None!!!
		ops = [model.test_support_acc]
		ops.extend(model.test_query_accs)
		result = sess.run(ops)
		test_accs.append(result)

	# [600, K+1]
	test_accs = np.array(test_accs)
	# [K+1]
	means = np.mean(test_accs, 0)
	stds = np.std(test_accs, 0)
	ci95 = 1.96 * stds / np.sqrt(600)

	print('[support_t0, query_t0 - \t\t\tK] ')
	print('mean:', means)
	print('stds:', stds)
	print('ci95:', ci95)



def main():
	training = not args.test
	kshot = 1
	kquery = 15
	nway = 5
	meta_batchsz = 4
	K = 5


	# kshot + kquery images per category, nway categories, meta_batchsz tasks.
	db = DataGenerator(nway, kshot, kquery, meta_batchsz, 200000)

	if  training:  # only construct training model if needed
		# get the tensor
		# image_tensor: [4, 80, 84*84*3]
		# label_tensor: [4, 80, 5]
		image_tensor, label_tensor = db.make_data_tensor(training=True)

		# NOTICE: the image order in 80 images should like this now:
		# [label2, label1, label3, label0, label4, and then repeat by 15 times, namely one task]
		# support_x : [4, 1*5, 84*84*3]
		# query_x   : [4, 15*5, 84*84*3]
		# support_y : [4, 5, 5]
		# query_y   : [4, 15*5, 5]
		support_x = tf.slice(image_tensor, [0, 0, 0], [-1,  nway *  kshot, -1], name='support_x')
		query_x = tf.slice(image_tensor, [0,  nway *  kshot, 0], [-1, -1, -1], name='query_x')
		support_y = tf.slice(label_tensor, [0, 0, 0], [-1,  nway *  kshot, -1], name='support_y')
		query_y = tf.slice(label_tensor, [0,  nway *  kshot, 0], [-1, -1, -1], name='query_y')

	# construct test tensors.
	image_tensor, label_tensor = db.make_data_tensor(training=False)
	support_x_test = tf.slice(image_tensor, [0, 0, 0], [-1,  nway *  kshot, -1], name='support_x_test')
	query_x_test = tf.slice(image_tensor, [0,  nway *  kshot, 0], [-1, -1, -1],  name='query_x_test')
	support_y_test = tf.slice(label_tensor, [0, 0, 0], [-1,  nway *  kshot, -1],  name='support_y_test')
	query_y_test = tf.slice(label_tensor, [0,  nway *  kshot, 0], [-1, -1, -1],  name='query_y_test')


	# 1. construct MAML model
	model = MAML(84, 3, 5)

	# construct metatrain_ and metaval_
	if  training:
		model.build(support_x, support_y, query_x, query_y, K, meta_batchsz, mode='train')
		model.build(support_x_test, support_y_test, query_x_test, query_y_test, K, meta_batchsz, mode='eval')
	else:
		model.build(support_x_test, support_y_test, query_x_test, query_y_test, K + 5, meta_batchsz, mode='test')
	model.summ_op = tf.summary.merge_all()

	all_vars = filter(lambda x: 'meta_optim' not in x.name, tf.trainable_variables())
	for p in all_vars:
		print(p)


	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)
	# tf.global_variables() to save moving_mean and moving variance of batch norm
	# tf.trainable_variables()  NOT include moving_mean and moving_variance.
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

	# initialize, under interative session
	tf.global_variables_initializer().run()
	tf.train.start_queue_runners()

	if os.path.exists(os.path.join('ckpt', 'checkpoint')):
		# alway load ckpt both train and test.
		model_file = tf.train.latest_checkpoint('ckpt')
		print("Restoring model weights from ", model_file)
		saver.restore(sess, model_file)


	if training:
		train(model, saver, sess)
	else:
		test(model, sess)


if __name__ == "__main__":
	main()
