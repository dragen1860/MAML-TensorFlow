import numpy as np
import tensorflow as tf


class MAML:
	def __init__(self, d, c, nway, meta_lr=1e-3, train_lr=1e-2):
		"""

		:param d:
		:param c:
		:param nway:
		:param meta_lr:
		:param train_lr:
		"""
		self.d = d
		self.c = c
		self.nway = nway
		self.meta_lr = meta_lr
		self.train_lr = train_lr

		print('img shape:', self.d, self.d, self.c, 'meta-lr:', meta_lr, 'train-lr:', train_lr)

	def build(self, support_xb, support_yb, query_xb, query_yb, K, meta_batchsz, mode='train'):
		"""

		:param support_xb:   [b, setsz, 84*84*3]
		:param support_yb:   [b, setsz, n-way]
		:param query_xb:     [b, querysz, 84*84*3]
		:param query_yb:     [b, querysz, n-way]
		:param K:           train update steps
		:param meta_batchsz:tasks number
		:param mode:        train/eval/test, for training, we build train&eval network meanwhile.
		:return:
		"""
		# create or reuse network variable, not including batch_norm variable, therefore we need extra reuse mechnism
		# to reuse batch_norm variables.
		self.weights = self.conv_weights()
		# TODO: meta-test is sort of test stage.
		training = True if mode is 'train' else False

		def meta_task(input):
			"""
			map_fn only support one parameters, so we need to unpack from tuple.
			:param support_x:   [setsz, 84*84*3]
			:param support_y:   [setsz, n-way]
			:param query_x:     [querysz, 84*84*3]
			:param query_y:     [querysz, n-way]
			:param training:    training or not, for batch_norm
			:return:
			"""
			support_x, support_y, query_x, query_y = input
			# to record the op in t update step.
			query_preds, query_losses, query_accs = [], [], []

			# ==================================
			# REUSE       True        False
			# Not exist   Error       Create one
			# Existed     reuse       Error
			# ==================================
			# That's, to create variable, you must turn off reuse
			support_pred = self.forward(support_x, self.weights, training)
			support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred, labels=support_y)
			support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(support_pred, dim=1), axis=1),
			                                             tf.argmax(support_y, axis=1))
			# compute gradients
			grads = tf.gradients(support_loss, list(self.weights.values()))
			# grad and variable dict
			gvs = dict(zip(self.weights.keys(), grads))

			# theta_pi = theta - alpha * grads
			fast_weights = dict(zip(self.weights.keys(),
			                        [self.weights[key] - self.train_lr * gvs[key] for key in self.weights.keys()]))
			# use theta_pi to forward meta-test
			query_pred = self.forward(query_x, fast_weights, training)
			# meta-test loss
			query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
			# record T0 pred and loss for meta-test
			query_preds.append(query_pred)
			query_losses.append(query_loss)

			# continue to build T1-TK steps graph
			for _ in range(1, K):
				# T_k loss on meta-train
				# we need meta-train loss to fine-tune the task and meta-test loss to update theta
				loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.forward(support_x, fast_weights, training),
				                                               labels=support_y)
				# compute gradients
				grads = tf.gradients(loss, list(fast_weights.values()))
				# compose grad and variable dict
				gvs = dict(zip(fast_weights.keys(), grads))
				# update theta_pi according to varibles
				fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.train_lr * gvs[key]
				                         for key in fast_weights.keys()]))
				# forward on theta_pi
				query_pred = self.forward(query_x, fast_weights, training)
				# we need accumulate all meta-test losses to update theta
				query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
				query_preds.append(query_pred)
				query_losses.append(query_loss)


			# compute every steps' accuracy on query set
			for i in range(K):
				query_accs.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(query_preds[i], dim=1), axis=1),
					                                            tf.argmax(query_y, axis=1)))
			# we just use the first step support op: support_pred & support_loss, but igonre these support op
			# at step 1:K-1.
			# however, we return all pred&loss&acc op at each time steps.
			result = [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]

			return result

		# return: [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]
		out_dtype = [tf.float32, tf.float32, tf.float32, [tf.float32] * K, [tf.float32] * K, [tf.float32] * K]
		result = tf.map_fn(meta_task, elems=(support_xb, support_yb, query_xb, query_yb),
		                   dtype=out_dtype, parallel_iterations=meta_batchsz, name='map_fn')
		support_pred_tasks, support_loss_tasks, support_acc_tasks, \
			query_preds_tasks, query_losses_tasks, query_accs_tasks = result


		if mode is 'train':
			# average loss
			self.support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
			# [avgloss_t1, avgloss_t2, ..., avgloss_K]
			self.query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
			                                        for j in range(K)]
			# average accuracy
			self.support_acc = support_acc = tf.reduce_sum(support_acc_tasks) / meta_batchsz
			# average accuracies
			self.query_accs = query_accs = [tf.reduce_sum(query_accs_tasks[j]) / meta_batchsz
			                                        for j in range(K)]

			# # add batch_norm ops before meta_op
			# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			# with tf.control_dependencies(update_ops):
			# 	# TODO: the update_ops must be put before tf.train.AdamOptimizer,
			# 	# otherwise it throws Not in same Frame Error.
			# 	meta_loss = tf.identity(self.query_losses[-1])

			# meta-train optim
			optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
			# meta-train gradients, query_losses[-1] is the accumulated loss across over tasks.
			gvs = optimizer.compute_gradients(self.query_losses[-1])
			# meta-train grads clipping
			gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
			# update theta
			self.meta_op = optimizer.apply_gradients(gvs)


		else: # test & eval

			# average loss
			self.test_support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
			# [avgloss_t1, avgloss_t2, ..., avgloss_K]
			self.test_query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
			                                        for j in range(K)]
			# average accuracy
			self.test_support_acc = support_acc = tf.reduce_sum(support_acc_tasks) / meta_batchsz
			# average accuracies
			self.test_query_accs = query_accs = [tf.reduce_sum(query_accs_tasks[j]) / meta_batchsz
			                                        for j in range(K)]

		# NOTICE: every time build model, support_loss will be added to the summary, but it's different.
		tf.summary.scalar(mode + '：support loss', support_loss)
		tf.summary.scalar(mode + '：support acc', support_acc)
		for j in range(K):
			tf.summary.scalar(mode + '：query loss, step ' + str(j + 1), query_losses[j])
			tf.summary.scalar(mode + '：query acc, step ' + str(j + 1), query_accs[j])




	def conv_weights(self):
		weights = {}
 
		conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
		fc_initializer = tf.contrib.layers.xavier_initializer()
		k = 3

		with tf.variable_scope('MAML', reuse= tf.AUTO_REUSE):
			weights['conv1']    = tf.get_variable('conv1w', [k, k, 3, 32],  initializer=conv_initializer)
			weights['b1']       = tf.get_variable('conv1b', initializer=tf.zeros([32]))
			weights['conv2']    = tf.get_variable('conv2w', [k, k, 32, 32], initializer=conv_initializer)
			weights['b2']       = tf.get_variable('conv2b', initializer=tf.zeros([32]))
			weights['conv3']    = tf.get_variable('conv3w', [k, k, 32, 32], initializer=conv_initializer)
			weights['b3']       = tf.get_variable('conv3b', initializer=tf.zeros([32]))
			weights['conv4']    = tf.get_variable('conv4w', [k, k, 32, 32], initializer=conv_initializer)
			weights['b4']       = tf.get_variable('conv4b', initializer=tf.zeros([32]))

			# assumes max pooling
			weights['w5']       = tf.get_variable('fc1w', [32 * 5 * 5, self.nway], initializer=fc_initializer)
			weights['b5']       = tf.get_variable('fc1b', initializer=tf.zeros([self.nway]))


			return weights

	def conv_block(self, x, weight, bias, scope, training):
		"""
		build a block with conv2d->batch_norm->pooling
		:param x:
		:param weight:
		:param bias:
		:param scope:
		:param training:
		:return:
		"""
		# conv
		x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME', name=scope + '_conv2d') + bias
		# batch norm, activation_fn=tf.nn.relu,
		# NOTICE: must have tf.layers.batch_normalization
		# x = tf.contrib.layers.batch_norm(x, activation_fn=tf.nn.relu)
		with tf.variable_scope('MAML'):
			# train is set to True ALWAYS, please refer to https://github.com/cbfinn/maml/issues/9
			# when FLAGS.train=True, we still need to build evaluation network
			x = tf.layers.batch_normalization(x, training=True, name=scope + '_bn', reuse=tf.AUTO_REUSE)
		# relu
		x = tf.nn.relu(x, name=scope + '_relu')
		# pooling
		x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name=scope + '_pool')
		return x


	def forward(self, x, weights, training):
		"""


		:param x:
		:param weights:
		:param training:
		:return:
		"""
		# [b, 84, 84, 3]
		x = tf.reshape(x, [-1, self.d, self.d, self.c], name='reshape1')

		hidden1 = self.conv_block(x,        weights['conv1'], weights['b1'], 'conv0', training)
		hidden2 = self.conv_block(hidden1,  weights['conv2'], weights['b2'], 'conv1', training)
		hidden3 = self.conv_block(hidden2,  weights['conv3'], weights['b3'], 'conv2', training)
		hidden4 = self.conv_block(hidden3,  weights['conv4'], weights['b4'], 'conv3', training)

		# get_shape is static shape, (5, 5, 5, 32)
		# print('flatten:', hidden4.get_shape())
		# flatten layer
		hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])], name='reshape2')

		output = tf.add(tf.matmul(hidden4, weights['w5']), weights['b5'], name='fc1')

		return output

