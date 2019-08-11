from PIL import Image

from os import path
import pickle, random

from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Scale(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		self.size = size
		self.interpolation = interpolation

	def __call__(self, img):
		return img.resize((self.size[1], self.size[0]), self.interpolation)

class VA_loader(Dataset):
	"""Data Generator for Valence Arousal Model"""

	def __init__(self, split, args):
		super(VA_loader, self).__init__()
		self.data_root = args.data_root
		self.img_dir = path.join(self.data_root, args.img_dir)
		self.split = split
		self.mode = args.mode
		self.config = args.config
		self.thresh = (args.thresh if split == 'training' else args.val_thresh)
		if type(self.thresh) == float:
			self.thresh = [self.thresh, self.thresh]

		self.data_prefix = args.prefix
		self.data = self.read_datapoints()

		#### first equally split train val across all emotion labels
		self.e2d = [[] for i in range(1, 8)]
		
		for i, d in enumerate(self.data):
			self.e2d[d['expression'] - 1].append(i)

		for i in range(len(self.e2d)):
			split_idx = int(.1 * len(self.e2d[i]))
			if self.split == 'training':
				self.e2d[i] = self.e2d[i][:-split_idx]
			else:
				self.e2d[i] = self.e2d[i][-split_idx:]

		self.data = [self.data[i] for i in sum(self.e2d,[])]

		print('Number of samples in {} : {}'.format(split, len(self.data)))

		self.aug_transforms = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(degrees=15.),
			transforms.ColorJitter(brightness=.1, contrast=.1, hue=.1, saturation=.1)
		])

		self.preprocess_transforms = transforms.Compose([
			Scale([args.img_size, args.img_size]),
			transforms.ToTensor(),
			transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
			std = [ 0.229, 0.224, 0.225 ]),
		])

	def read_datapoints(self):
		# reads datapoints from {split}.csv and stores resized faces 
		if path.exists('{}{}{}_data'.format(self.data_root, self.data_prefix, self.split)):
			return (pickle.load(open(path.join(self.data_root, self.data_prefix + 'training_data'), 'rb')) + \
					pickle.load(open(path.join(self.data_root, self.data_prefix + 'validation_data'), 'rb')))

		ann_file = path.join(self.data_root, '{}.csv'.format(self.split))

		df = pd.read_csv(ann_file, delimiter=',')

		labels = range(1, 8)
		# remove unwanted labels
		df = df[[x in labels for x in df.expression]] 

		# modify image path to image name
		df.subDirectory_filePath = df.subDirectory_filePath.apply(lambda x : x.split('/')[1])
		
		# remove absent images
		df = df[[path.exists(path.join(self.img_dir, x)) for x in df.subDirectory_filePath]]
		
		data = []
		for i in range(len(df)):
			attrs = list(df.iloc[i, :])
			attrs = {attr_name : attrs[i] for i, attr_name in enumerate(['fname'] + list(df.columns)[1:])}

			data.append(attrs)

			# one last check
			assert data[i]['valence'] > -1.5
			assert data[i]['arousal'] > -1.5
			assert data[i]['expression'] < 8

		pickle.dump(data, open('{}{}_data'.format(self.data_root, self.split), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
		return data

	def get_item_by_id(self, idx):
		fname, valence, arousal = (self.data[idx]['fname'], 
									self.data[idx]['valence'], 
									self.data[idx]['arousal'])

		face = Image.open(path.join(self.img_dir, fname)).convert('RGB')

		if self.split == 'training':
			face = self.aug_transforms(face)

		return face, valence, arousal

	def __getitem__(self, idx):
		np.random.seed(random.randint(0, 1000000))

		def _get_label(x, y, thresh):
			if np.abs(x - y) < thresh:
				z = 1
			elif x > y:
				z = 0
			else:
				z = 2
			return z
		
		face, valence, arousal = self.get_item_by_id(idx)
		
		if self.config == 'ranking':
			id2 = np.random.choice(len(self.data))
			face2, valence2, arousal2 = self.get_item_by_id(id2)

			return face, face2, _get_label(valence, valence2, self.thresh[0]), _get_label(arousal, arousal2, self.thresh[1])

		return face, valence, arousal


	def __len__(self):
		return len(self.data)

	def dump_resized(self, size, faces_dir=):
		from concurrent.futures import ThreadPoolExecutor, wait, as_completed
		from multiprocessing import cpu_count
		pool = ThreadPoolExecutor(2 * cpu_count())

		def _resize_and_save(idx):
			fname = self.data[idx]['fname']
			if path.isfile(faces_dir + fname):
				return

			img = Image.open(path.join(self.img_dir, fname)).convert('RGB')
			cropx, cropy, width, height = (self.data[idx]['face_x'],
											self.data[idx]['face_y'],
											self.data[idx]['face_width'],
											self.data[idx]['face_height'])
			face = img.crop((cropx, cropy, cropx + width, cropy + height))

			face = face.resize((size, size), Image.BILINEAR)

			face.save(faces_dir + fname)

		futures = [pool.submit(_resize_and_save, idx) for idx in range(len(self.data))]
		_ = [r.result() for r in tqdm(as_completed(futures), total=len(self.data))]