import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import sys
import math
import time


if __name__ == '__main__':
	params = {}
	params['batch_size'] = 10
	params['learning_rate'] = 0.01
	# params['layer1_filter_size'] = 5
	# params['layer1_depth'] = 16
	# params['layer1_stride'] = 2
	# params['layer2_filter_size'] = 5
	# params['layer2_depth'] = 16
	# params['layer2_stride'] = 2
	# params['layer3_num_hidden'] = 64
	# params['layer4_num_hidden'] = 64
	params['num_training_steps'] = 1501

	layer1 = {}
	layer2 = {}
	layer3 = {}
	layer4 = {}
	layer1['filter_size'] = 5
	layer1['depth'] = 16
	layer1['stride'] = 2
	layer1['pooling'] = False
	layer1['hidden'] = False
	layer2['filter_size'] = 5
	layer2['depth'] = 16
	layer2['stride'] = 2
	layer2['pooling'] = False
	layer2['hidden'] = False
	layer3['hidden'] = True
	layer3['num_hidden'] = 64
	layer4['hidden'] = True
	layer4['num_hidden'] = 64

	params['layers'] = [layer1, layer2, layer3, layer4]

	for count, l in enumerate(params['layers']):
		# print l
		# l['weights'] = 3
		#print l
		print l
		print count
		if count > 0:
			print params['layers'][count-1]

	#print params['layers'][1]['pooling']
