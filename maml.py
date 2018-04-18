import numpy as np
import tensorflow as tf
from tensorflow import flags

FLAGS = flags.FLAGS


class MAML:
	def __init__(self, dim_input, dim_output, test_num_updates=5):
		"""

		:param dim_input:
		:param dim_output:
		:param test_num_updates:
		"""
		self.dim_input = dim_input
		self.dim_output = dim_output
		self.test_num_updates = test_num_updates
		self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())

		self.img_size = int(np.sqrt(self.dim_input / 3))

	def build(self, input, prefix='metatrain_'):
		"""

		:param input:
		:param prefix:
		:return:
		"""
		# support_x : [4, 1*5, 84*84*3]
		# query_x   : [4, 15*5, 84*84*3]
		# support_y : [4, 5, 5]
		# query_y   : [4, 15*5, 5]
		self.support_x  = input['support_x']
		self.support_y  = input['support_y']
		self.query_x    = input['query_x']
		self.query_y    = input['query_y']

		# train iteration
		K = max(self.test_num_updates, FLAGS.train_iteration)
		# num of tasks
		N = tf.to_float(FLAGS.meta_batchsz)

		with tf.variable_scope('MAML', reuse=None) as scope:
			# since we need to construct train model and test model, this function will be used for twice
			if 'weights' in dir(self):
				scope.reuse_variables()
				weights = self.weights
			else:
				# build the weights
				self.weights = weights = self.conv_weights()

			# at this time, we have weigths and self.weights
			# outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates

			# the following list save all tasks' op.
			support_pred_tasks, support_loss_tasks, support_acc_tasks = [], [], []
			query_preds_tasks, query_losses_tasks, query_accs_tasks = [[]] * K, [[]] * K, [[]] * K

			def meta_task(input, reuse=True):
				""" Perform gradient descent for one task in the meta-batch. """
				support_x, query_x, support_y, query_y = input
				# to record the op in t update step.
				query_preds, query_losses, query_accs = [], [], []

				# forward: support_x -> 4conv -> fc -> [5]
				support_pred = self.forward(support_x, weights, reuse=reuse)  # only reuse on the first iter
				support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred, labels=support_y)
				# compute gradients
				grads = tf.gradients(support_loss, list(weights.values()))
				gvs = dict(zip(weights.keys(), grads))

				# theta_pi = theta - alpha * grads
				fast_weights = dict(zip(weights.keys(), [weights[key] - FLAGS.train_lr * gvs[key] for key in weights.keys()]))
				# use theta_pi to forward
				query_pred = self.forward(query_x, fast_weights, reuse=True)
				query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
				query_preds.append(query_pred)
				query_losses.append(query_loss)

				for j in range(K - 1):
					loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.forward(support_x, fast_weights, reuse=True),
					                                               labels=support_y)
					# compute gradients
					grads = tf.gradients(loss, list(fast_weights.values()))
					# compose grad and variable dict
					gvs = dict(zip(fast_weights.keys(), grads))
					# update theta_pi according to varibles
					fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - FLAGS.train_lr * gvs[key]
					                         for key in fast_weights.keys()]))
					# forward on theta_pi
					query_pred = self.forward(query_x, fast_weights, reuse=True)
					query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
					query_preds.append(query_pred)
					query_losses.append(query_loss)


				support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(support_pred), 1),
				                                             tf.argmax(support_y, 1))
				for j in range(K):
					query_accs.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(query_preds[j]), 1),
						                                            tf.argmax(query_y, 1)))
				# we just use the first step support op: support_pred & support_loss, but igonre these support op
				# at step 1:K-1.
				# however, we return all pred&loss&acc op at each time steps.
				result = [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]

				return result

			# to initialize the batch norm variables, might want to combine this, and not run idx 0 twice.
			unused = meta_task((self.support_x[0], self.query_x[0], self.support_y[0], self.query_y[0]), False)

			# support_x : [4, 1*5, 84*84*3]
			# query_x   : [4, 15*5, 84*84*3]
			# support_y : [4, 5, 5]
			# query_y   : [4, 15*5, 5]
			# return: [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]
			out_dtype = [tf.float32, tf.float32, tf.float32, [tf.float32] * K, [tf.float32] * K, [tf.float32] * K]
			support_pred_tasks, support_loss_tasks, support_acc_tasks, \
				query_preds_tasks, query_losses_tasks, query_accs_tasks = \
					tf.map_fn(meta_task, elems=(self.support_x, self.query_x, self.support_y, self.query_y),
			                   dtype=out_dtype,
			                   parallel_iterations=FLAGS.meta_batchsz)


		## Performance & Optimization
		if 'train' in prefix:

			# no need to average
			self.support_pred_tasks, self.query_preds_tasks = support_pred_tasks, query_preds_tasks

			# average loss
			self.support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / N
			# [avgloss_t1, avgloss_t2, ..., avgloss_K]
			self.query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / N
			                                        for j in range(K)]
			# average accuracy
			self.support_acc = support_acc = tf.reduce_sum(support_acc_tasks) / N
			# average accuracies
			self.query_accs = query_accs = [tf.reduce_sum(query_accs_tasks[j]) / N
			                                        for j in range(K)]
			# inner-train op
			self.train_op = tf.train.AdamOptimizer(self.meta_lr).minimize(support_loss)
			# meta-train optim
			optimizer = tf.train.AdamOptimizer(self.meta_lr)
			# meta-train gradients
			self.gvs = gvs = optimizer.compute_gradients(self.query_losses[K - 1])
			# meta-train grads clipping
			gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
			# update theta
			self.meta_op = optimizer.apply_gradients(gvs)

		else: # test

			# average loss
			self.test_support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / N
			# [avgloss_t1, avgloss_t2, ..., avgloss_K]
			self.test_query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / N
			                                        for j in range(K)]
			# average accuracy
			self.test_support_acc = support_acc = tf.reduce_sum(support_acc_tasks) / N
			# average accuracies
			self.test_query_accs = query_accs = [tf.reduce_sum(query_accs_tasks[j]) / N
			                                        for j in range(K)]

		## Summaries
		# NOTICE: every time build model, support_loss will be added to the summary, but it's different.
		tf.summary.scalar(prefix + 'support loss', support_loss)
		tf.summary.scalar(prefix + 'support acc', support_acc)
		for j in range(K):
			tf.summary.scalar(prefix + 'query loss, step ' + str(j + 1), query_losses[j])
			tf.summary.scalar(prefix + 'query acc, step ' + str(j + 1), query_accs[j])




	def conv_weights(self):
		weights = {}
 
		conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
		fc_initializer = tf.contrib.layers.xavier_initializer()
		k = 3

		weights['conv1']    = tf.get_variable('conv1', [k, k, 3, 32],  initializer=conv_initializer)
		weights['b1']       = tf.Variable(tf.zeros([32]))
		weights['conv2']    = tf.get_variable('conv2', [k, k, 32, 32], initializer=conv_initializer)
		weights['b2']       = tf.Variable(tf.zeros([32]))
		weights['conv3']    = tf.get_variable('conv3', [k, k, 32, 32], initializer=conv_initializer)
		weights['b3']       = tf.Variable(tf.zeros([32]))
		weights['conv4']    = tf.get_variable('conv4', [k, k, 32, 32], initializer=conv_initializer)
		weights['b4']       = tf.Variable(tf.zeros([32]))


		# assumes max pooling
		weights['w5']       = tf.get_variable('w5', [32 * 5 * 5, FLAGS.nway], initializer=fc_initializer)
		weights['b5']       = tf.Variable(tf.zeros([self.dim_output]), name='b5')


		return weights

	def conv_block(self, x, weight, bias, reuse, scope):
		"""
		build a block with conv2d->batch_norm->pooling
		:param x:
		:param weight:
		:param bias:
		:param reuse:
		:param scope:
		:return:
		"""
		# conv
		x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME') + bias
		# batch norm, activation_fn=tf.nn.relu,
		x = tf.contrib.layers.batch_norm(x, activation_fn=tf.nn.relu, reuse=reuse, scope=scope)
		# pooling
		x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
		return x


	def forward(self, x, weights, reuse=False, scope='conv'):
		"""


		:param x:
		:param weights:
		:param reuse: reuse is for the normalization parameters.
		:param scope:
		:return:
		"""
		# [b, 84, 84, 3]
		x = tf.reshape(x, [-1, self.img_size, self.img_size, 3])

		hidden1 = self.conv_block(x,        weights['conv1'], weights['b1'], reuse, scope + '0')
		hidden2 = self.conv_block(hidden1,  weights['conv2'], weights['b2'], reuse, scope + '1')
		hidden3 = self.conv_block(hidden2,  weights['conv3'], weights['b3'], reuse, scope + '2')
		hidden4 = self.conv_block(hidden3,  weights['conv4'], weights['b4'], reuse, scope + '3')

		# get_shape is static shape, (5, 5, 5, 32)
		# print('flatten:', hidden4.get_shape())
		# flatten layer
		hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])

		output = tf.matmul(hidden4, weights['w5']) + weights['b5']

		return output
