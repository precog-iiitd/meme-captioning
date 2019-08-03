import glob
import math
import numpy as np
import os
import os.path as osp
import string 
import pickle
import json
from time import sleep
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Scale(object):
  """Scale transform with list as size params"""

  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation

  def __call__(self, img):
    return img.resize((self.size[1], self.size[0]), self.interpolation)

def get_vocab():
  worddict_tmp = pickle.load(open('data/wordlist.p', 'rb'))
  wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
  wordlist = ['EOS'] + sorted(wordlist)
  return wordlist

def get_basic_transforms(img_size=128):
  resize_transform = transforms.Compose([Scale([img_size, img_size]),])
  img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
      std = [ 0.229, 0.224, 0.225 ]),
    ])
  
  return resize_transform, img_transforms

class coco_loader(Dataset):
  """Loads train/val/test splits of coco dataset"""

  def __init__(self, data_root, split='train', max_tokens=15, ncap_per_img=3, img_size=128):
    self.max_tokens = max_tokens
    self.ncap_per_img = ncap_per_img
    self.data_root = data_root
    self.split = split
    #Splits from http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    self.get_split_info('data/dataset.json')

    self.wordlist = get_vocab()

    self.numwords = len(self.wordlist)
    print('[DEBUG] #words in wordlist: %d' % (self.numwords))

    self.resize_transform, self.img_transforms = get_basic_transforms(img_size)

    self.aug_transforms = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(degrees=15),
      transforms.ColorJitter(brightness=.1, contrast=.1, hue=.1, saturation=.1)
    ])

  def get_split_info(self, split_file):
    print('Loading annotation file...')
    with open(split_file) as fin:
      split_info = json.load(fin)
    annos = {}
    labels = {}
    for item in split_info['images']:
      if self.split == 'train':
        if item['split'] == 'train' or item['split'] == 'restval':
          annos[item['cocoid']] = item
      elif item['split'] == self.split:
        annos[item['cocoid']] = item
        labels[item['cocoid']] = item['labels']

    self.annos = annos
    self.labels = labels
    self.ids = list(self.annos.keys())
    print('Found %d images in split: %s'%(len(self.ids), self.split))

  def __getitem__(self, idx):
    img_id = self.ids[idx]
    anno = self.annos[img_id]

    captions = [caption['raw'] for caption in anno['sentences']]

    imgpath = '%s/%s/%s'%(self.data_root, anno['filepath'], anno['filename'])
    img = Image.open(os.path.join(imgpath)).convert('RGB')

    img = self.resize_transform(img)
    
    if self.split == 'train':
      img = self.aug_transforms(img)

    img = self.img_transforms(img)

    if(self.split != 'train'):
      r = np.random.randint(0, len(captions))
      captions = [captions[r]]

    if (self.split == 'train'):
      if(len(captions) > self.ncap_per_img):
        ids = np.random.permutation(len(captions))[:self.ncap_per_img]
        captions_sel = [captions[l] for l in ids]
        captions = captions_sel
      assert(len(captions) == self.ncap_per_img)

    wordclass = torch.LongTensor(len(captions), self.max_tokens).zero_()
    sentence_mask = torch.ByteTensor(len(captions), self.max_tokens).zero_()

    for i, caption in enumerate(captions):
      words = str(caption).lower().translate(None, string.punctuation).strip().split()
      words = ['<S>'] + words
      num_words = min(len(words), self.max_tokens-1)
      sentence_mask[i, :(num_words+1)] = 1
      for word_i, word in enumerate(words):
        if(word_i >= num_words):
          break
        if(word not in self.wordlist):
          word = 'UNK'
        wordclass[i, word_i] = self.wordlist.index(word)

    return img, captions, wordclass, sentence_mask, img_id 

  def __len__(self):
    return len(self.ids)

def get_glove_vectors(wordlist, size=300):
  np.random.seed(0)
  print ("Loading Glove Model")
  f = open('data/glove.6B.{}d.txt'.format(size),'r')

  glove = {}
  for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    glove[word] = embedding
  print ("Done. {} words loaded!".format(len(glove)))

  word_embeddings = [np.zeros(size)]
  loaded = 1

  for w in wordlist[1:]:
    if w in ['UNK', '<S>']:
      word_embeddings.append(np.zeros(size, dtype=np.float32))
      loaded += 1
    elif w in glove:
      word_embeddings.append(glove[w])
      loaded += 1
    else:
      word_embeddings.append(np.random.random(size))
  print('{} / {} words intialized with glove!'.format(loaded, len(wordlist)))

  return word_embeddings