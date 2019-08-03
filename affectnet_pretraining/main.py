import argparse, os
from tqdm import tqdm
import numpy as np

from datagen import VA_loader
from models import VA, VA_multiple, VA_separate, separate, VA_branch

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch Valence Arousal Model')

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--base', type=str, default='resnet50')
parser.add_argument('--pool', type=str, default='avg')

parser.add_argument('--config', type=str, default='ranking', help='ranking (default) | regression')
parser.add_argument('--thresh', type=list, default=[.25, .25])
parser.add_argument('--val_thresh', type=list, default=[.25, .25])

parser.add_argument('--model_dir', type=str, default='models/')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--data_root', type=str, default='data/')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--repeat_val', type=int, default=20)

parser.add_argument('--img_dir', type=str, default='faces/', help='Image folder')
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('-se', '--start_epoch', type=int, default=0)
parser.add_argument('-e', '--epochs', type=int, default=1000)
parser.add_argument('-b', '--batchsize', type=int, default=128)
parser.add_argument('-m', '--nthreads', type=int, default=6,\
					help='pytorch data loader threads')
parser.add_argument('-lr', '--lr', type=float, default=0.1)
parser.add_argument('-loss', '--loss_fn', type=str, default='mse')
parser.add_argument('-opt', '--optimizer', type=str, default='sgd')
parser.add_argument('-schedule', '--lr_schedule', type=str, default='on_plateau')

args = parser.parse_args()

def get_lr(e):
	base = args.lr
	if e < 5: 
		return args.lr
	elif e < 10:
		return args.lr / 5
	elif e < 15:
		return args.lr / 10
	return args.lr / 20

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)
	
def acc(predicted, actual):
	predicted_labels = np.argmax(predicted, axis=1).reshape(-1)
	actual = actual.reshape(-1).astype(np.uint8)

	return sum(predicted_labels == actual) / float(len(actual))

def custom_mse_loss(predicted, actual, thresh):
	return torch.mean(torch.clamp(torch.abs(predicted - actual) - thresh, min=0.))

def regression_acc(predicted, actual, thresh):
	return float(sum(np.abs(predicted - actual).reshape(-1) < thresh)) / len(actual)

def run_epoch_regression(model, data, num_batches, args, prog=True):
	loss, a_acc, v_acc = 0., 0., 0.

	if prog:
		data_prog_bar = tqdm(enumerate(data), total=num_batches)
	else:
		data_prog_bar = enumerate(data)

	for batch_idx, (imgs, v, a) in data_prog_bar:
		args.optimizer.zero_grad()
		batchsize = len(imgs)
		imgs = Variable(imgs.view(batchsize, 3, args.img_size, args.img_size)).cuda()

		v = v.view(batchsize, 1).type(torch.FloatTensor).cuda()
		a = a.view(batchsize, 1).type(torch.FloatTensor).cuda()

		pred_v, pred_a = model(imgs)

		batch_loss_valence = args.loss_fn(pred_v, v, args.thresh[0])
		batch_loss_arousal = args.loss_fn(pred_a, a, args.thresh[1])

		batch_loss = .5 * (batch_loss_valence + batch_loss_arousal)

		a_acc += regression_acc(pred_a.data.cpu().numpy(), a.data.cpu().numpy(), args.thresh[0])
		v_acc += regression_acc(pred_v.data.cpu().numpy(), v.data.cpu().numpy(), args.thresh[1])

		if data.dataset.split == 'training':
			batch_loss.backward()
			args.optimizer.step()

		loss += batch_loss.data.item()

		if prog:
			disp_loss = round(loss / (batch_idx + 1), 4)
			disp_a_acc = round(a_acc / (batch_idx + 1), 4)
			disp_v_acc = round(v_acc / (batch_idx + 1), 4)

			data_prog_bar.set_description('L : {} ; a : {} ; v : {}'.format(disp_loss, disp_a_acc, disp_v_acc))

	return (loss * 1.) / (batch_idx + 1), a_acc / (batch_idx + 1), v_acc / (batch_idx + 1)

def run_epoch_ranking(model, data, num_batches, args, prog=True):
	loss, a_acc, v_acc = 0., 0., 0.

	if prog:
		data_prog_bar = tqdm(enumerate(data), total=num_batches)
	else:
		data_prog_bar = enumerate(data)

	for batch_idx, (img1, img2, v, a) in data_prog_bar:
		args.optimizer.zero_grad()
		batchsize = len(img1)

		v = v.view(-1).type(torch.LongTensor).cuda()
		a = a.view(-1).type(torch.LongTensor).cuda()
		img1, img2 = img1.cuda(), img2.cuda()

		pred_v, pred_a = model(img1, img2)

		batch_loss_valence = args.loss_fn['valence'](pred_v, v)
		batch_loss_arousal = args.loss_fn['arousal'](pred_a, a)

		batch_loss = .5 * (batch_loss_valence + batch_loss_arousal)

		if data.dataset.split == 'training':
			batch_loss.backward()
			args.optimizer.step()

		pred_a, pred_v = torch.softmax(pred_a, dim=1), torch.softmax(pred_v, dim=1)

		a_acc += acc(pred_a.data.cpu().numpy(), a.data.cpu().numpy())
		v_acc += acc(pred_v.data.cpu().numpy(), v.data.cpu().numpy())

		loss += batch_loss.data.item()

		if prog:
			disp_loss = round(loss / (batch_idx + 1), 4)
			disp_a_acc = round(a_acc / (batch_idx + 1), 4)
			disp_v_acc = round(v_acc / (batch_idx + 1), 4)

			data_prog_bar.set_description('L : {} ; a : {} ; v : {}'.format(disp_loss, disp_a_acc, disp_v_acc))

	return (loss * 1.) / (batch_idx + 1), (a_acc * 1.) / (batch_idx + 1), (v_acc * 1.) / (batch_idx + 1)

def train(args):
	train_data = VA_loader('training', args)
	val_data = VA_loader('validation', args)

	num_train_batches = int(np.floor((len(train_data.data)*1.)/args.batchsize))
	num_val_batches = int(np.floor((len(val_data.data)*1.)/args.batchsize))

	train_data_loader = DataLoader(dataset=train_data, num_workers=args.nthreads,\
								batch_size=args.batchsize, shuffle=True, drop_last=True)
	val_data_loader = DataLoader(dataset=val_data, num_workers=args.nthreads,\
								batch_size=args.batchsize, shuffle=True, drop_last=False)
	print('Data files loaded')

	if args.config == 'ranking':
		model = VA_ranking()
	elif args.config == 'regression':
		model = VA_regression()
	else:
		raise ValueError('Invalid config')

	model.cuda()
	model.train(True)
	print('Model defined')
	print('Total number of trainable parameters: {}'.format(count_parameters(model)))

	if args.optimizer == 'sgd':
		args.optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr,
							momentum=.9, nesterov=True, weight_decay=1e-4)
	elif args.optimizer == 'adam':
		args.optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], weight_decay=1e-4)
	else:
		raise NotImplementedError(args.optimizer)

	lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(args.optimizer, factor=.5, patience=1, 
												threshold=0.02, threshold_mode='abs', min_lr=1e-4, verbose=True)
	train_history, val_history = [], []

	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch'] + 1
			model.load_state_dict(checkpoint['state_dict'])
			args.optimizer.load_state_dict(checkpoint['optimizer'])
			train_history, val_history = checkpoint['train_history'], checkpoint['val_history']

			print("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']))
		else:
			raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

	if args.config == 'ranking':
		# v=.2 , [.29, .42, .29] && a=.3 , [.42, .16, .42]
		# v=.2 , [.29, .42, .29] && a=.2 , [.37, .26, .37]
		args.loss_fn = {'valence' : nn.CrossEntropyLoss(),#weight=torch.FloatTensor([.35, .3, .35]).cuda()), 
						'arousal' : nn.CrossEntropyLoss()}#weight=torch.FloatTensor([.4, .2, .4]).cuda())}
	elif args.loss_fn == 'mse':
		args.loss_fn = custom_mse_loss #nn.MSELoss()
	else:
		raise NotImplementedError(args.loss_fn)

	best_val_loss = np.inf

	regular_checkpoint_path = os.path.join(args.model_dir, 'model_{}_{}.pth')
	best_checkpoint_path = os.path.join(args.model_dir, 'best_{}_{}.pth')

	if args.config == 'ranking':
		runner = run_epoch_ranking
	else:
		runner = run_epoch_regression

	for epoch in range(args.start_epoch, args.epochs):
		model.train(True)

		train_loss, _, _ = runner(model, train_data_loader, num_train_batches, args)
		print('Training epoch {} has loss {}'.format(epoch, train_loss))

		torch.save({
			'epoch': epoch,
			'state_dict': model.state_dict(),
			'optimizer' : args.optimizer.state_dict(),
			'train_history':train_history,
			'val_history':val_history
		}, regular_checkpoint_path.format(epoch, train_loss))
		train_history.append(train_loss)

		lr_scheduler.step(train_loss)

		model.eval()
		repeat_val = args.repeat_val
		val_loss, v_acc, a_acc = [], [], []
		with torch.no_grad():
			prog_bar = tqdm(range(repeat_val))
			for r in prog_bar:
				vl, aa, va = runner(model, val_data_loader, num_val_batches, args, prog=False)
				val_loss.append(vl)
				v_acc.append(va)
				a_acc.append(aa)
				disp_loss = round(np.mean(val_loss), 4)
				disp_a_acc = round(np.mean(a_acc), 4)
				disp_v_acc = round(np.mean(v_acc), 4)

				prog_bar.set_description('L : {} ; a : {} ; v : {}'.format(disp_loss, disp_a_acc, disp_v_acc))

		print('Validation epoch {} has loss {}'.format(epoch, np.mean(val_loss)))

		val_loss = np.mean(val_loss)
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			print('Validation loss at epoch {} is: {}'.format(epoch, val_loss))
			os.system('cp {} {}'.format(regular_checkpoint_path.format(epoch, train_loss), 
									best_checkpoint_path.format(epoch, val_loss)))

		val_history.append(val_loss)

		print('Train history: {}'.format(train_history))
		print('Test history: {}'.format(val_history))

if __name__ == '__main__':
	if args.mode == 'train':
		train(args)
	elif args.mode == 'preprocess':
		train_data = VA_loader('training', args)
		val_data = VA_loader('validation', args)

		train_data.dump_resized(size=args.img_size, faces_dir=os.path.join(args.data_root, 'faces'))
		val_data.dump_resized(size=args.img_size, faces_dir=os.path.join(args.data_root, 'faces'))
	else:
		raise ValueError('Not Implemented mode {} yet'.format(args.mode))