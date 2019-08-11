from __future__ import print_function

import os
import os.path as osp
import argparse

from train import train 
from test import test
  
parser = argparse.ArgumentParser(description='PyTorch Convolutional Image Captioning Model')

parser.add_argument('-mode', '--mode', type=str, default='test', help='train | test (default)')

parser.add_argument('--model_dir', default='saved_models/', help='output directory to save models & results')

parser.add_argument('--checkpoint', default=None, help='checkpoint path to load weights')
parser.add_argument('--imgcnn', default=None, help='imgcnn_checkpoint path' 
                    'will be overridden by --checkpoint parameter if the latter is specified ')

parser.add_argument('--data_root', type=str, default= 'data/', help='Root directory where splits created by'
                    'prepare_splits.py are stored')
parser.add_argument('--img_size', type=int, default=128,\
                    help='Image size')

parser.add_argument('-e', '--epochs', type=int, default=1000000,\
                    help='number of training epochs')

parser.add_argument('-b', '--batchsize', type=int, default=3,\
                    help='number of images per training batch')

parser.add_argument('-c', '--ncap_per_img', type=int, default=3,\
                    help='ground-truth captions per image in training batch')

parser.add_argument('-n', '--num_layers', type=int, default=3,\
                    help='depth of convcap network')

parser.add_argument('-m', '--nthreads', type=int, default=4,\
                    help='pytorch data loader threads')

parser.add_argument('-ft', '--finetune_after', type=int, default=100000000000,\
                    help='epochs after which imgcnn is fine-tuned')

parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5,\
                    help='learning rate for convcap')
parser.add_argument('-gc', '--grad_clip', type=float, default=.1,\
                    help='Clip gradient by ? value, default=.1')

parser.add_argument('-sc', '--score_select', type=str, default='CIDEr',\
                    help='metric to pick best model')

parser.add_argument('--attention', dest='attention', default=True, \
                    help='Use this for convcap with attention (default: True)')

parser.add_argument('-glove', '--glove', type=int, default=1, help='Use pretrained glove embeddings?')
parser.add_argument('-ge', '--ge', type=int, default=300, help='Glove embedding dim, default: 300')
parser.add_argument('-train_embeddings', '--train_embeddings', type=bool, default=False, 
                  help='To train or freeze word embeddings, default: False')

parser.add_argument('-interval', '--interval', type=int, default=5, help='Save checkpoint every ? epochs')

args = parser.parse_args()


def main():
  if args.mode == 'train':
    train(args)
  elif args.mode == 'test':
    if(osp.exists(args.checkpoint)):
      scores = test(args, 'holdout', modelfn=args.checkpoint)

      print('TEST set scores')
      for k, v in scores[0].iteritems():
        print('%s: %f' % (k, v))
    else:
      raise IOError('No checkpoint found %s' % args.checkpoint)
  else:
    raise ValueError('Incorrect mode!!')

if __name__ == '__main__': 
  main()
