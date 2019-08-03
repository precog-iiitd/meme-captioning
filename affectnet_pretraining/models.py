from resnet import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F

class VA_regression(nn.Module):
	"""
	Basic Valence Arousal model

	Arguments: 
		base 				: CNN encoder to get a set of feature maps
		num_feature_maps	: Number of feature maps in the output
	"""
	def __init__(self):
		super(VA, self).__init__()
		self.base = resnet50()

		num_feature_maps = self.base.num_feature_maps
		self.valence = nn.Sequential(*[nn.Linear(num_feature_maps, 1)])
		self.arousal = nn.Sequential(*[nn.Linear(num_feature_maps, 1)])

	def forward(self, img):
		features = self.base(img)
		pooled = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
		v, a = self.valence(pooled), self.arousal(pooled)

		return v, a

class VA_ranking(nn.Module):
	"""
	Basic Valence Arousal model

	Arguments: 
		base 				: CNN encoder to get a set of feature maps
		num_feature_maps	: Number of feature maps in the output
	"""
	def __init__(self):
		super(VA_multiple, self).__init__()
		self.base = resnet50()
		
		num_feature_maps = self.base.num_feature_maps

		self.valence = nn.Sequential(*[nn.Linear(2 * num_feature_maps, 3)])
		self.arousal = nn.Sequential(*[nn.Linear(2 * num_feature_maps, 3)])

	def forward(self, img1, img2):
		features1 = self.base(img1)
		features2 = self.base(img2)

		pooled1, pooled2 = features1, features2
		feat = torch.cat([pooled1, pooled2], dim=1)
		v, a = self.valence(feat), self.arousal(feat)

		return v, a
