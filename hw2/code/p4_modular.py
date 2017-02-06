import numpy as np
from plotBoundary import *
import pylab as pl
import gradient_homebrew as gh
import scipy.optimize
import time
from utils import * #my functions for HW2
# import your LR training code

if __name__=='__main__':
	# parameters
	Name = [('1','7'),('3','5'),('4','9')]
	Gamma = [2e-2, 2e-1, 2e0, 2e1, 2e2];
	C = [0.01, 0.1, 1, 10, 100]

	#Gamma = [2e0];
	#C = [1]
	# load data from csv files

	# Name = ['4']
	# Gamma = [2e0];
	# C = [1.0]

	EM = np.zeros([len(Name),len(Gamma),len(C),2])
	margin = np.zeros([len(Name),len(Gamma),len(C),1])
	NSV = np.zeros([len(Name),len(Gamma),len(C),1])
	for ni,name in enumerate(Name):
		for gj,gamma in enumerate(Gamma):
			for ck,c in enumerate(C):
				print '======Training======'
				#train = np.loadtxt('data/data'+name+'_train.csv')
				# use deep copy here to make cvxopt happy
				#train = np.loadtxt('data/data'+name+'_train.csv')
				train1 = np.loadtxt('data/mnist_digit_'+name[0]+'.csv')
				train7 = np.loadtxt('data/mnist_digit_'+name[1]+'.csv')

				n = 200
				X1 = train1[:n,:]
				X7 = train7[:n,:]
				X = np.vstack((X1,X7))
				#normalize
				X = 2*X/255-1
				#print(X.shape)
				tmp1 = np.ones(n)
				tmp2 = np.ones(n)*-1
				Y = np.hstack((tmp1,tmp2))
				#print(Y.shape)

				K = rbf(gamma,X)
				n,m = K.shape

				alphai,Xi,Yi,b = train_kQPSVM(X,Y,K,c,gamma)

				margin[ni,gj,ck,0] = 1/np.linalg.norm(alphai)
				NSV[ni,gj,ck,0] = alphai.shape[0]
				predictSVM = constructGaussianPredictorB(alphai,gamma,Xi,Yi,b)

				corr = 0.0
				for i in range(n):
					corr = corr + np.abs(Y[i]-np.sign(predictSVM(X[i,:])))
				print(corr/2/X.shape[0])
				EM[ni,gj,ck,0] = corr/2/X.shape[0]

				print '======Validation======'
				vn = 150
				X1v = train1[n:n+vn,:]
				X7v = train7[n:n+vn,:]
				Xv = np.vstack((X1v,X7v))
				#normalize
				Xv = 2*Xv/255-1
				tmp1 = np.ones(vn)	
				tmp2 = np.ones(vn)*-1
				Yv = np.hstack((tmp1,tmp2))

				n, m = Xv.shape
				corr = 0.0
				for i in range(n):
					corr = corr + np.abs(Yv[i]-np.sign(predictSVM(Xv[i,:])))
				print(corr/2/Xv.shape[0])
				EM[ni,gj,ck,1] = corr/2/X.shape[0]

	np.save('EM_gaussian_4.txt',EM)
	np.save('margin_gaussian_4.txt',margin)
	np.save('NSV_gaussian_4.txt',NSV)