# MAML-TensorFlow
An elegant and efficient implementation for ICML2017 paper: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)

# Highlights
- adopted from cbfin's official implementation with equivalent performance on mini-imagenet
- clean, tiny code style and very easy-to-follow from comments almost every lines
- faster and trivial improvements, eg. 0.335s per epoch comparing with 0.563s per epoch, saving up to **3.8 hours** for total 60,000 training process


# How-TO
1. Download mini-Imagenet from [here](https://drive.google.com/open?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk) and extract them as :
```shell
	miniimagenet/	
	├── images	
		├── n0210891500001298.jpg  		
		├── n0287152500001298.jpg 		
		...		
	├── test.csv	
	├── val.csv	
	└── train.csv	
	└── proc_images.py
	
```

then replace the `path` by your actual path in `data_generator.py`:
```python
		metatrain_folder = config.get('metatrain_folder', '/hdd1/liangqu/datasets/miniimagenet/train')
		if True:
			metaval_folder = config.get('metaval_folder', '/hdd1/liangqu/datasets/miniimagenet/test')
		else:
			metaval_folder = config.get('metaval_folder', '/hdd1/liangqu/datasets/miniimagenet/val')
```	

2. resize all raw images to 84x84 size by
```shell
python proc_images.py
```

3. train
```shell
python main.py
```
Since tf.summary is time-consuming, I turn it off by default.
uncomment the 2 lines to turn it on:
```python
	# write graph to tensorboard
	# tb = tf.summary.FileWriter(os.path.join('logs', 'mini'), sess.graph)

	...
	# summ_op
	# tb.add_summary(result[1], iteration)
```
and then minitor training process by:
```shell
tensorboard --logdir logs
```

4. test
```shell
python main.py --test
```

As MAML need generate 200,000 train/eval episodes before training, which usually takes up to 6~8 minutes, I use an cache file `filelist.pkl` to dump all these episodes for the first time and then next time the program will load from the cached file. It only takes several seconds to load from cached files.

generating episodes: 100%|█████████████████████████████████████████| 200000/200000 [04:38<00:00, 717.27it/s]

