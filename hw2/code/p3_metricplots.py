import numpy as np
import numpy.matlib
from plotBoundary import *
import pylab as pl
from itertools import compress
from scipy.spatial.distance import cdist,pdist,squareform
from utils import *

if __name__=='__main__':
	# parameters
	Name = ['1','2','3','4']
	Gamma = [2e-2, 2e-1, 2e0, 2e1, 2e2];
	Lmbda_gp = [0.02]
	Lmbda_g = [2, 2e-1, 2e-2, 2e-3, 2e-4, 2e-5, 2e-6, 2e-7, 2e-8, 2e-9, 2e-10]
	C = [0.01, 0.1, 1, 10, 100]

	EM_gp = np.load('EM_gaussian_p.txt.npy')
	margin_gp = np.load('margin_gaussian_p.txt.npy')
	NSV_gp = np.load('NSV_gaussian_p.txt.npy')
	print(EM_gp.shape)
	print(margin_gp.shape)
	print(NSV_gp.shape)

	EM_lp = np.load('EM_linear_p.txt.npy')
	margin_lp = np.load('margin_linear_p.txt.npy')
	NSV_lp = np.load('NSV_linear_p.txt.npy')
	print(EM_lp.shape)
	print(margin_lp.shape)
	print(NSV_lp.shape)

	EM_g = np.load('EM_gaussian.txt.npy')
	margin_g = np.load('margin_gaussian.txt.npy')
	NSV_g = np.load('NSV_gaussian.txt.npy')
	print(EM_g.shape)
	print(margin_g.shape)
	print(NSV_g.shape)

	EM_l = np.load('EM_linear.txt.npy')
	margin_l = np.load('margin_linear.txt.npy')
	NSV_l = np.load('NSV_linear.txt.npy')
	print(EM_l.shape)
	print(margin_l.shape)
	print(NSV_l.shape)

	print(EM_l.T)
	print(margin_l.T)
	print(NSV_l.T)


	# print(margin_g.shape)
	# for ni,name in enumerate(Name):
	# 	for nc,c in enumerate(C):
	# 		pl.loglog(Gamma,margin_g[ni,:,nc,0])

	# pl.show()
	# pl.close()

	# print(NSV_l.shape)
	# for ni,name in enumerate(Name):
	# 	for nc,c in enumerate(C):
	# 		pl.loglog(Gamma,NSV_l[ni,:,nc,0])

	# pl.show()
	# pl.close()

	# print(EM_l.shape)
	# #for ni,name in enumerate(Name):
	# for nc,c in enumerate(C):
	# 	pl.bar([1,2,3,4],EM_l[:,nc,1])
	# 	pl.show()
	
	pl.close()
