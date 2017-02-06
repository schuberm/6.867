import time
import numpy as np
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
import load_datasets as ld
import pickle
import numpy.matlib
from scipy.optimize import minimize, fmin_cobyla
from random import randint
from joblib import Parallel, delayed

class uap:

	def __init__(self,loss):
		self.loss = loss

	def predictor(self,loss):
		def predict(x):
			return numpy.asscalar(numpy.argmax(loss(x), axis=1))
		return predict

	# def uap_parallel(self,data):
	# 	np = 4
	# 	ms = 16
	# 	results = Parallel(n_jobs=-1,backend="multiprocessing", verbose=5)((delayed(uapcalc)( self, data[ii*int(ms/np):(ii+1)*int(ms/np),:])) for ii in range(np))
	# 	return results