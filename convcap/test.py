import os
import os.path as osp
import argparse
import numpy as np 
import json
import time
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm 
 
from data_loader import coco_loader, get_glove_vectors
from torchvision import models                                                                     
from convcap import convcap
from evaluate import language_eval
from resnet import resnet50, ResNet, rename_keys

def repeat_img_per_cap(imgsfeats, imgsfc7, ncap_per_img):
  """Repeat image features ncap_per_img times"""

  batchsize, featdim, feat_h, feat_w = imgsfeats.size()
  batchsize_cap = batchsize*ncap_per_img
  imgsfeats = imgsfeats.unsqueeze(1).expand(\
    batchsize, ncap_per_img, featdim, feat_h, feat_w)
  imgsfeats = imgsfeats.contiguous().view(\
    batchsize_cap, featdim, feat_h, feat_w)
  
  batchsize, featdim = imgsfc7.size()
  batchsize_cap = batchsize*ncap_per_img
  imgsfc7 = imgsfc7.unsqueeze(1).expand(\
    batchsize, ncap_per_img, featdim)
  imgsfc7 = imgsfc7.contiguous().view(\
    batchsize_cap, featdim)

  return imgsfeats, imgsfc7

def test(args, split, modelfn=None, model_convcap=None, model_imgcnn=None):
  """Runs test on split=val/test with checkpoint file modelfn or loaded model_*"""

  t_start = time.time()
  data = coco_loader(args.data_root, split=split, ncap_per_img=1)
  print('[DEBUG] Loading %s data ... %f secs' % (split, time.time() - t_start))

  data_loader = DataLoader(dataset=data, num_workers=args.nthreads,\
    batch_size=args.batchsize, shuffle=False, drop_last=False)

  batchsize = args.batchsize
  max_tokens = data.max_tokens
  num_batches = np.int_(np.floor((len(data.ids)*1.)/batchsize))
  if num_batches == 0: num_batches = 1

  print('[DEBUG] Running inference on %s with %d batches' % (split, num_batches))

  model_imgcnn = resnet50().cuda()

  model_imgcnn.load_state_dict(rename_keys(torch.load(modelfn)['state_dict']))

  word_embeddings = None
  if args.glove:
    word_embeddings = get_glove_vectors(args.ge, data.wordlist)
    
  model_convcap = convcap(data.numwords, args.num_layers, is_attention=args.attention,
                         embedding_weights=word_embeddings).cuda()

  if modelfn is not None:
    print('[DEBUG] Loading checkpoint %s' % modelfn)
    checkpoint = torch.load(modelfn)
    model_convcap.load_state_dict(checkpoint['state_dict'])
    model_imgcnn.load_state_dict(checkpoint['img_state_dict'])

  model_imgcnn.eval() 
  model_convcap.eval()

  pred_captions = []
  attns = []
  pred_tokens = []
  all_img_ids = []
  loss = 0.

  for batch_idx, (imgs, _, wordclass_t, mask, img_ids) in \
    tqdm(enumerate(data_loader), total=num_batches):
    batchsize = len(imgs)
    wordact_t_final = [None for _ in range(batchsize * (max_tokens - 1))]

    imgs = imgs.view(batchsize, 3, 128, 128)

    imgs_v = Variable(imgs.cuda())
    imgsfeats, imgsfc7 = model_imgcnn(imgs_v)

    _, featdim, feat_h, feat_w = imgsfeats.size()
  
    wordclass_feed = np.zeros((batchsize, max_tokens), dtype='int64')
    wordclass_feed[:,0] = data.wordlist.index('<S>') 

    outcaps = np.empty((batchsize, 0)).tolist()

    for j in range(max_tokens-1):
      wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()

      wordact, attn = model_convcap(imgsfeats, imgsfc7, wordclass)
      if args.mode == 'attvis':
        attn = attn.view(batchsize, max_tokens, feat_h, feat_w)

      wordact = wordact[:,:,:-1]
      wordact_t = wordact.permute(0, 2, 1).contiguous().view(batchsize*(max_tokens-1), -1)

      wordprobs = F.softmax(wordact_t).cpu().data.numpy()
      wordids = np.argmax(wordprobs, axis=1)

      for k in range(batchsize):
        word = data.wordlist[wordids[j+k*(max_tokens-1)]]
        outcaps[k].append(word)
        if(j < max_tokens-1):
          wordclass_feed[k, j+1] = wordids[j+k*(max_tokens-1)]

        wordact_t_final[j + k * (max_tokens-1)] = wordact_t[j + k * (max_tokens-1)]
    
    wordclass_t = wordclass_t.view(batchsize, max_tokens)
    mask = mask.view(batchsize, max_tokens)
    wordclass_t = wordclass_t[:,1:]
    mask = mask[:,1:].contiguous()
    wordact_t_final = torch.stack(wordact_t_final).cpu()
    wordclass_t = wordclass_t.contiguous().view(batchsize*(max_tokens-1), 1)
    maskids = torch.nonzero(mask.view(-1)).numpy().reshape(-1)

    loss += F.cross_entropy(wordact_t_final[maskids, ...], \
      wordclass_t[maskids, ...].contiguous().view(maskids.shape[0])).data.item()

    for j in range(batchsize):
      num_words = len(outcaps[j]) 
      if 'EOS' in outcaps[j]:
        num_words = outcaps[j].index('EOS')
      outcap = ' '.join(outcaps[j][:num_words])
      if args.mode == 'attvis':
        pred_tokens.append(outcaps[j][:num_words])
      pred_captions.append({'image_id': img_ids[j].item(), 'caption': outcap})

  print('{} split testing loss is: {}'.format(split, (loss*1.)/(batch_idx + 1)))

  scores = language_eval(pred_captions, args.model_dir, split)
  if args.mode == 'test':
    labelnames = ['Happy', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Sad']
    label_wise_caps = {labelnames[i] : [p for p in pred_captions if data.labels[p['image_id']][i]]\
                       for i in range(6)}
    for k, preds in label_wise_caps.iteritems():
      print(k)
      language_eval(preds, args.model_dir, split)
      
  return scores 
 
