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
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision import models                                                                     

from data_loader import coco_loader, get_glove_vectors
from convcap import convcap
from tqdm import tqdm 
from test import test
from resnet import resnet50, ResNet, rename_keys

def count_parameters(model):
  print('\n'.join(['{} \t {}'.format(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]))
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def train(args):
  t_start = time.time()
  train_data = coco_loader(args.data_root, split='train', ncap_per_img=args.ncap_per_img, img_size=args.img_size)
  print('[DEBUG] Loading train data ... %f secs' % (time.time() - t_start))

  train_data_loader = DataLoader(dataset=train_data, num_workers=args.nthreads,\
    batch_size=args.batchsize, shuffle=True, drop_last=True)

  model_imgcnn = resnet50().cuda()
  model_imgcnn.eval() # freeze weights by default

  if args.imgcnn != None:
    print('Loading imgcnn pre-trained model: {}'.format(args.imgcnn))
    try:
      model_imgcnn.load_state_dict(rename_keys(torch.load(args.imgcnn)['state_dict']))
    except RuntimeError:
      model_imgcnn.load_state_dict(torch.load(args.imgcnn)['state_dict'])

  print('{}training image CNN'.format('' if model_imgcnn.training else 'Not '))

  if args.glove:
    word_embeddings = get_glove_vectors(args.ge, train_data.wordlist)

  #Convcap model
  model_convcap = convcap(train_data.numwords, args.num_layers, is_attention=args.attention, 
                          embedding_weights=word_embeddings, train_embeddings=args.train_embeddings).cuda()
  model_convcap.train(True)

  if args.checkpoint != None:
    print('Resuming from : {}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    model_convcap.load_state_dict(checkpoint['state_dict'])
    model_imgcnn.load_state_dict(checkpoint['img_state_dict'])

  print('Number of trainable params in ConvCap: {}'.format(count_parameters(model_convcap)))

  optimizer = optim.RMSprop([p for p in model_convcap.parameters() if p.requires_grad], lr=args.learning_rate)
  img_optimizer = None

  batchsize = args.batchsize
  ncap_per_img = args.ncap_per_img
  batchsize_cap = batchsize*ncap_per_img
  max_tokens = train_data.max_tokens
  nbatches = np.int_(np.floor((len(train_data.ids)*1.)/batchsize)) 
  bestscore = .0
  os.makedirs(args.model_dir, exist_ok=True)
  train_history, val_history = [], []

  for epoch in range(args.epochs):
    loss_train = 0.
    model_convcap.train()
    if epoch >= args.finetune_after:
      model_imgcnn.train()
      if not img_optimizer:
        model_imgcnn.train()
        for name, p in list(model_imgcnn.named_parameters()):
          if 'layer4' not in name:
            p.requires_grad = False

        print('Training imgcnn now!')
        print(sum([p.requires_grad for p in model_imgcnn.parameters()]), 
            'layers out of {} are set to trainable'.format(len(list(model_imgcnn.parameters()))))
        img_optimizer = optim.RMSprop([p for p in model_imgcnn.parameters() if p.requires_grad], 
                                      lr=args.learning_rate / 5)

    #One epoch of train
    for batch_idx, (imgs, captions, wordclass, mask, _) in \
      tqdm(enumerate(train_data_loader), total=nbatches):

      imgs = imgs.view(batchsize, 3, args.img_size, args.img_size)
      wordclass = wordclass.view(batchsize_cap, max_tokens)
      mask = mask.view(batchsize_cap, max_tokens)

      imgs_v = Variable(imgs).cuda()
      wordclass_v = Variable(wordclass).cuda()

      optimizer.zero_grad()
      if(img_optimizer):
        img_optimizer.zero_grad() 

      imgsfeats, imgsfc7 = model_imgcnn(imgs_v)

      imgsfeats, imgsfc7 = repeat_img_per_cap(imgsfeats, imgsfc7, ncap_per_img)
      _, _, feat_h, feat_w = imgsfeats.size()

      if args.head:
        feat_h, feat_w = 8, 8

      if(args.attention):
        wordact, attn = model_convcap(imgsfeats, imgsfc7, wordclass_v)
        attn = attn.view(batchsize_cap, max_tokens, feat_h, feat_w)
      else:
        wordact, _ = model_convcap(imgsfeats, imgsfc7, wordclass_v)

      wordact = wordact[:,:,:-1]
      wordclass_v = wordclass_v[:,1:]
      mask = mask[:,1:].contiguous()

      wordact_t = wordact.permute(0, 2, 1).contiguous().view(\
        batchsize_cap*(max_tokens-1), -1)
      wordclass_t = wordclass_v.contiguous().view(\
        batchsize_cap*(max_tokens-1), 1)
      
      maskids = torch.nonzero(mask.view(-1)).numpy().reshape(-1)

      if(args.attention):
        #Cross-entropy loss and attention loss of Show, Attend and Tell
        loss = F.cross_entropy(wordact_t[maskids, ...], \
          wordclass_t[maskids, ...].contiguous().view(maskids.shape[0])) \
          + (torch.sum(torch.pow(1. - torch.sum(attn, 1), 2)))\
          /(batchsize_cap*feat_h*feat_w)
      else:
        loss = F.cross_entropy(wordact_t[maskids, ...], \
          wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))

      loss_train = loss_train + loss.data.item()

      loss.backward()
      nn.utils.clip_grad_value_([p for p in model_convcap.parameters() if p.requires_grad], args.gc)

      optimizer.step()
      if(img_optimizer):
        img_optimizer.step()

    loss_train = (loss_train*1.)/(batch_idx + 1)
    print('[DEBUG] Training epoch %d has loss %f' % (epoch, loss_train))

    modelfn = osp.join(args.model_dir, 'model.pth')

    if(img_optimizer):
      img_optimizer_dict = img_optimizer.state_dict()
    else:
      img_optimizer_dict = None

    torch.save({
        'epoch': epoch,
        'state_dict': model_convcap.state_dict(),
        'img_state_dict': model_imgcnn.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'img_optimizer' : img_optimizer_dict,
      }, modelfn)

    #Run on validation and obtain score
    with torch.no_grad():
      if epoch != 0 and epoch % args.interval == 0:
        scores = test(args, 'train', model_convcap=model_convcap, model_imgcnn=model_imgcnn)
        train_history.append(scores[0][args.score_select])
        scores = test(args, 'test', model_convcap=model_convcap, model_imgcnn=model_imgcnn)  
        val_history.append(scores[0][args.score_select])
        score = scores[0][args.score_select]

        if(score > bestscore):
          bestscore = score
          print('[DEBUG] Saving model at epoch %d with %s score of %f'\
            % (epoch, args.score_select, score))
          bestmodelfn = osp.join(args.model_dir, 'bestmodel.pth')
          os.system('cp %s %s' % (modelfn, bestmodelfn))

        print('Train history: {}'.format(train_history))
        print('Test history: {}'.format(val_history))

  