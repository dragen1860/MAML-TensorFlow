import numpy as np
import os
import random
import tensorflow as tf

from tensorflow import flags
from utils import get_images

FLAGS = flags.FLAGS


class DataGenerator:
	"""
	Data Generator capable of generating batches of sinusoid or Omniglot data.
	A "class" is considered a class of omniglot digits or a particular sinusoid function.
	"""

	def __init__(self, num_samples_per_class, batch_size, config={}):
		"""
		Args:
			num_samples_per_class: num samples to generate per class in one batch
			batch_size: size of meta batch size (e.g. number of functions)
		"""
		self.batch_size = batch_size
		self.num_samples_per_class = num_samples_per_class
		self.num_classes = 1  # by default 1 (only relevant for classification problems)

		self.num_classes = config.get('num_classes', FLAGS.nway)
		self.img_size = config.get('img_size', (84, 84))
		self.dim_input = np.prod(self.img_size) * 3 # 21168
		self.dim_output = self.num_classes
		metatrain_folder = config.get('metatrain_folder', '../maml/data/miniImagenet/train')
		if True:
			metaval_folder = config.get('metaval_folder', '../maml/data/miniImagenet/test')
		else:
			metaval_folder = config.get('metaval_folder', '../maml/data/miniImagenet/val')

		metatrain_folders = [os.path.join(metatrain_folder, label) \
		                     for label in os.listdir(metatrain_folder) \
		                     if os.path.isdir(os.path.join(metatrain_folder, label)) \
		                     ]
		metaval_folders = [os.path.join(metaval_folder, label) \
		                   for label in os.listdir(metaval_folder) \
		                   if os.path.isdir(os.path.join(metaval_folder, label)) \
		                   ]
		self.metatrain_character_folders = metatrain_folders
		self.metaval_character_folders = metaval_folders
		self.rotations = config.get('rotations', [0])


		print('metatrain_folder', metatrain_folders[:5])
		print('metaval_folders', metaval_folders[:5])


	def make_data_tensor(self, train=True):
		if train:
			folders = self.metatrain_character_folders
			# number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
			num_total_batches = 2000
		else:
			folders = self.metaval_character_folders
			num_total_batches = 600

		# make list of files
		print('Generating filenames')
		# 16 in one class, 16*5 in one task
		# [task1_0_img0, task1_0_img15, task1_1_img0,]
		all_filenames = []
		for _ in range(num_total_batches): # 200000
			# from image folder sample 5 class randomly
			sampled_character_folders = random.sample(folders, self.num_classes)
			random.shuffle(sampled_character_folders)
			# sample 16 images from selected folders, and each with label 0-4, (0/1..., path)
			# len: 5 * 16
			labels_and_images = get_images(sampled_character_folders, range(self.num_classes),
			                               nb_samples=self.num_samples_per_class, shuffle=False)

			# make sure the above isn't randomized order
			labels = [li[0] for li in labels_and_images]
			filenames = [li[1] for li in labels_and_images]
			all_filenames.extend(filenames)

		# make queue for tensorflow to read from
		print('Generating image processing ops')
		filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
		image_reader = tf.WholeFileReader()
		_, image_file = image_reader.read(filename_queue)


		image = tf.image.decode_jpeg(image_file, channels=3)
		# tensorflow format: N*H*W*C
		image.set_shape((self.img_size[0], self.img_size[1], 3))
		# reshape(image, [84*84*3])
		image = tf.reshape(image, [self.dim_input])
		# convert to range(0,1)
		image = tf.cast(image, tf.float32) / 255.0


		num_preprocess_threads = 1  # TODO - enable this to be set to >1
		min_queue_examples = 256
		examples_per_batch = self.num_classes * self.num_samples_per_class # 5*16
		# batch here means batch of meta-learning, including 4 tasks = 4*80
		batch_image_size = self.batch_size * examples_per_batch # 4* 80

		print('Batching images')
		images = tf.train.batch(
			[image],
			batch_size=batch_image_size, # 4*80
			num_threads=num_preprocess_threads, # 1
			capacity=min_queue_examples + 3 * batch_image_size,
		)

		all_image_batches, all_label_batches = [], []
		print('Manipulating image data to be right shape')
		# images contains current batch, namely 4 task, 4* 80
		for i in range(self.batch_size): # 4
			# current task, 80 images
			image_batch = images[i * examples_per_batch:(i + 1) * examples_per_batch]

			# as all labels of all task are the same, which is 0,0,..1,1,..2,2,..3,3,..4,4...
			label_batch = tf.convert_to_tensor(labels)
			new_list, new_label_list = [], []
			# for each image from 0 to 15 in all 5 class
			for k in range(self.num_samples_per_class): # 16
				class_idxs = tf.range(0, self.num_classes) # 0-4
				class_idxs = tf.random_shuffle(class_idxs)
				# it will cope with 5 images parallelly
				#    [0, 16, 32, 48, 64] or [1, 17, 33, 49, 65]
				true_idxs = class_idxs * self.num_samples_per_class + k
				new_list.append(tf.gather(image_batch, true_idxs))

				new_label_list.append(tf.gather(label_batch, true_idxs))

			# [80, 84*84*3]
			new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
			# [80]
			new_label_list = tf.concat(new_label_list, 0)
			all_image_batches.append(new_list)
			all_label_batches.append(new_label_list)

		# [4, 80, 84*84*3]
		all_image_batches = tf.stack(all_image_batches)
		# [4, 80]
		all_label_batches = tf.stack(all_label_batches)
		# [4, 80, 5]
		all_label_batches = tf.one_hot(all_label_batches, self.num_classes)

		print('image_b:', all_image_batches)
		print('label_onehot_b:', all_label_batches)

		return all_image_batches, all_label_batches

