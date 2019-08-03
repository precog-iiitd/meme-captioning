import numpy as np
import json, pickle, sys, argparse, os
from nltk import word_tokenize
from shutil import copy
import pandas as pd

def preprocess(d):
	d = d.strip().lower()
	if d[-1] == '.': d = d[:-1] # remove full-stop
	return d

dictionary = {'UNK' : 0, '<S>': 1}
def add_to_dict(word):
	global dictionary
	if word not in dictionary:
		dictionary[word] = len(dictionary)

def read_descriptions(args, include_ann_ids=['ann_0', 'ann_1', 'ann_2']):
	fname = os.path.join(args.dataset_dir, 'descriptions.json')
	data = json.load(open(fname))['data']

	imgid = 0
	sentid = 0
	dataset = {'images':[], 'dataset':'coco'} # simply following coco-style

	for sample in data:
		fname, anns = sample['fname'], sample['anns']

		valid_anns = [anns[ann_id] for ann_id in include_ann_ids if ann_id in anns]
		if len(valid_anns) != len(include_ann_ids): continue
		
		item = {}
		item['filename'] = fname
		item['labels'] = sample['labels']
		item['imgid'] = item['cocoid'] = imgid
		item['sentences'] = []
		item['sentids'] = []

		for ann in valid_anns:
			description = {}

			ann = preprocess(ann)
			description['raw'] = ann
			description['tokens'] = word_tokenize(ann)
			description['sentid'] = sentid

			item['sentences'].append(description)
			item['sentids'].append(sentid)

			for word in description['tokens']: add_to_dict(word)

			sentid += 1

		dataset['images'].append(item)
		imgid += 1

	return dataset

parser = argparse.ArgumentParser(description='Split the dataset into train, test, holdout splits')
parser.add_argument('--dataset_dir', default='../dataset/', type=str, help='Path to dataset root')
parser.add_argument('--num_test', default=192, type=int, help='No. of samples in test split')
parser.add_argument('--seed', default=0, type=int, help='Seed value')

args = parser.parse_args()

if __name__ == '__main__':
	if not os.path.isdir('third_party/coco-caption'):
		raise IOError('Please download and install coco-caption in'
					' the correct directory before running this script')

	dataset = read_descriptions(args)
	num_samples = len(dataset['images'])
	print('Dataset contains : {} samples.'.format(num_samples))

	num_val, num_test = args.num_test, args.num_test
	
	np.random.seed(args.seed)
	split = np.random.permutation([0]*(num_samples - num_test - num_val) +\
							 [1]*num_test + [2]*num_val) # 1 = test, 0 = train
	
	splitnames = ['train', 'holdout', 'test']
	for splitname in splitnames: os.makedirs(os.path.join('data', splitname), exist_ok=True)

	for i in range(len(dataset['images'])):
		dataset['images'][i]['split'] = dataset['images'][i]['filepath'] = splitnames[split[i]]
		copy(os.path.join(args.dataset_dir, 'final_faces', dataset['images'][i]['filename']), 
			os.path.join('data', dataset['images'][i]['filepath'], dataset['images'][i]['filename']))

	print('Split into : {} train and {} test'.format(num_samples - 2 * num_test, 2 * num_test))
	json.dump(dataset, open('data/dataset.json', 'w'))
	print('Dumping vocabulary of {} words'.format(len(dictionary)))
	pickle.dump(dictionary, open('data/wordlist.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


	# for coco-metrics evaluation

	info = {
				"year" : 2019,
				"version" : '1',
				"description" : 'CaptionEval',
				"contributor" : 'Prajwal',
				"url" : 'https://gitlab.com/prajwalkr/',
				"date_created" : '',
				}
	licenses = [{
				"id" : 1,
				"name" : "test",
				"url" : "test",
				}]
	
	images = []
	annotations = []
	for item in dataset['images']:
		image_dict = {"id" : item['cocoid'],
					"width" : 0,
					"height" : 0,
					"file_name" : item['filename'],
					"license" : '',
					"url" : item['filename'],
					"date_captured" : '',
					}
		images.append(image_dict)

		for ann in item['sentences']:
			ann_dict = { "id" : ann['sentid'],
						"image_id" : item['cocoid'],
						"caption" : ann['raw']}
						
			annotations.append(ann_dict)

	res = {"info" : info,
			"type" : 'captions',
			"images" :  images,
			"annotations" : annotations,
			"licenses" : licenses,
		}

	json.dump(res, open('third_party/coco-caption/annotations/eval.json', 'w'))	
