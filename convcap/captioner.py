import argparse, torch, pickle, os
from data_loader import get_vocab, get_basic_transforms, get_glove_vectors
from PIL import Image
from resnet import resnet50, ResNet, rename_keys
from convcap import convcap
from beamsearch import beamsearch
from torch.autograd import Variable
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def caption(args):
	img = Image.open(args.img).convert('RGB')
	resize_transform, img_transforms = get_basic_transforms()

	img = img_transforms(resize_transform(img))

	wordlist = get_vocab()
	word_embeddings_cache_path = 'data/vocab_embeddings.p'
	if os.path.isfile(word_embeddings_cache_path):
		with open(word_embeddings_cache_path, 'rb') as f:
			word_embeddings = pickle.load(f)
	else:
		word_embeddings = get_glove_vectors(wordlist)
		with open(word_embeddings_cache_path, 'wb') as f:
			pickle.dump(word_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

	model_imgcnn = resnet50().cuda()
	model_imgcnn.eval()

	model_convcap = convcap(len(wordlist), args.num_layers, embedding_weights=word_embeddings).cuda()
	model_convcap.eval()

	print('[DEBUG] Loading checkpoint %s' % args.checkpoint)
	checkpoint = torch.load(args.checkpoint)
	model_convcap.load_state_dict(checkpoint['state_dict'])
	model_imgcnn.load_state_dict(checkpoint['img_state_dict'])

	pred_captions = []
	img_v = Variable(img.cuda())
	imgfeats, imgfc7 = model_imgcnn(img_v.unsqueeze(0))

	b, f_dim, f_h, f_w = imgfeats.size()
	imgfeats = imgfeats.unsqueeze(1).expand(\
	  b, args.beam_size, f_dim, f_h, f_w)
	imgfeats = imgfeats.contiguous().view(\
	  b*args.beam_size, f_dim, f_h, f_w)

	b, f_dim = imgfc7.size()
	imgfc7 = imgfc7.unsqueeze(1).expand(\
	  b, args.beam_size, f_dim)
	imgfc7 = imgfc7.contiguous().view(\
	  b*args.beam_size, f_dim)

	batchsize = 1
	max_tokens = 15
	beam_searcher = beamsearch(args.beam_size, batchsize, max_tokens)
	
	wordclass_feed = np.zeros((args.beam_size*batchsize, max_tokens), dtype='int64')
	wordclass_feed[:,0] = wordlist.index('<S>') 
	outcaps = np.empty((batchsize, 0)).tolist()

	for j in range(max_tokens-1):
	  wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()

	  wordact, attn = model_convcap(imgfeats, imgfc7, wordclass)
	  wordact = wordact[:,:,:-1]
	  wordact_j = wordact[..., j]

	  beam_indices, wordclass_indices = beam_searcher.expand_beam(wordact_j)  

	  if len(beam_indices) == 0 or j == (max_tokens-2): # Beam search is over.
	    generated_captions = beam_searcher.get_results()
	    for k in range(batchsize):
	        g = generated_captions[:, k]
	        outcaps[k] = [wordlist[x] for x in g]
	  else:
	    wordclass_feed = wordclass_feed[beam_indices]
	    imgfc7 = imgfc7.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
	    imgfeats = imgfeats.index_select(0, Variable(torch.cuda.LongTensor(beam_indices)))
	    for i, wordclass_idx in enumerate(wordclass_indices):
	      wordclass_feed[i, j+1] = wordclass_idx

	outcaps = outcaps[0]
	num_words = len(outcaps) 
	if 'EOS' in outcaps:
	  num_words = outcaps.index('EOS')
	outcap = ' '.join(outcaps[:num_words])
	print(outcap)
	return outcap

parser = argparse.ArgumentParser(description='Caption a face image using a trained model')

parser.add_argument('--img', help='checkpoint path to load weights')
parser.add_argument('--checkpoint', help='checkpoint path to load weights')

# model params

parser.add_argument('-n', '--num_layers', type=int, default=3,\
                    help='depth of convcap network')
parser.add_argument('-b', '--beam_size', type=int, default=1,\
                    help='Beam size for prediction')

args = parser.parse_args()


if __name__ == '__main__':
	caption(args)