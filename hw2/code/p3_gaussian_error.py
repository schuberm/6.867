import numpy as np
import numpy.matlib
from plotBoundary import *
import pylab as pl
from itertools import compress
from scipy.spatial.distance import cdist,pdist,squareform
from utils import *
# import your LR training code

if __name__=='__main__':
	# parameters
	Name = ['1','2','3','4']
	Gamma = [2e-2, 2e-1, 2e0, 2e1, 2e2];
	Lmbda = [0.02]

	# Name = ['4']
	# Gamma = [2e0];
	# Lmbda = [0.02]
	epochs = 1000;

	EM = np.zeros([len(Name),len(Gamma),len(Lmbda),3])
	margin = np.zeros([len(Name),len(Gamma),len(Lmbda),1])
	NSV = np.zeros([len(Name),len(Gamma),len(Lmbda),1])
	for ni,name in enumerate(Name):
		for gj,gamma in enumerate(Gamma):
			for lk,lmbda in enumerate(Lmbda):
				print '======Training======'
				# load data from csv files
				train = np.loadtxt('data/data'+name+'_train.csv')
				# use deep copy here to make cvxopt happy
				X = train[:, 0:2].copy()
				Y = train[:, 2:3].copy()
				n,m = X.shape

				K = rbf(gamma,X)
				### TODO: Implement train_gaussianSVM ###
				alphai, Xi, Yi, error = train_kPegasosSVM(X, Y, K, lmbda, epochs)
				pl.plot(error)
				pl.xlabel('epoch')
				pl.ylabel('Error')
				pl.savefig('./figures/ps33_error.pdf')
				input()
				margin[ni,gj,lk,0] = 1/np.linalg.norm(alphai)
				NSV[ni,gj,lk,0] = alphai.shape[0]
				predict_gaussianSVM = constructGaussianPredictor(alphai,gamma,Xi,Yi)

				corr = 0.0
				for i in range(n):
					corr = corr + np.abs(Y[i]-np.sign(predict_gaussianSVM (X[i,:])))
				print(corr/2/X.shape[0])
				EM[ni,gj,lk,0] = corr/2/X.shape[0]
				
				plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel Pegasos SVM')
				#pl.show()
				pl.savefig('./figures/ps33_training'+str(name)+'_'+str(gamma)+'_'+str(lmbda)+'.pdf')
				pl.close()
				print '======Validation======'
				# load data from csv files
				validate = np.loadtxt('./data/data'+name+'_validate.csv')
				X = validate[:, 0:2]
				Y = validate[:, 2:3]
				# plot validation results
				n, m = X.shape
				corr = 0.0
				for i in range(n):
					corr = corr + np.abs(Y[i]-np.sign(predict_gaussianSVM (X[i,:])))
				print(corr/2/X.shape[0])

				plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel Pegasos SVM')
				#pl.show()
				pl.savefig('./figures/ps33_validate'+str(name)+'_'+str(gamma)+'_'+str(lmbda)+'.pdf')
				pl.close()
				EM[ni,gj,lk,1] = corr/2/X.shape[0]

				print '======Testing======'
				# load data from csv files
				validate = np.loadtxt('./data/data'+name+'_test.csv')
				X = validate[:, 0:2]
				Y = validate[:, 2:3]
				# plot validation results
				n, m = X.shape
				corr = 0.0
				for i in range(n):
					corr = corr + np.abs(Y[i]-np.sign(predict_gaussianSVM (X[i,:])))
				print(corr/2/X.shape[0])

				plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel Pegasos SVM')
				#pl.show()
				pl.savefig('./figures/ps33_test'+str(name)+'_'+str(gamma)+'_'+str(lmbda)+'.pdf')
				pl.close()
				EM[ni,gj,lk,2] = corr/2/X.shape[0]

	#print(EM)
	# np.save('EM_gaussian_p.txt',EM)
	# np.save('margin_gaussian_p.txt',margin)
	# np.save('NSV_gaussian_p.txt',NSV)
