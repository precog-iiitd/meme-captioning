import json
from PIL import Image

dataset = json.load(open('dataset.json'))['images']

for d in dataset:
	fname = d['filename']
	split = d['split']
	Image.open(split + '/' + fname).show()
	sentences = d['sentences']
	for s in sentences:
		print s['raw']

	raw_input()

