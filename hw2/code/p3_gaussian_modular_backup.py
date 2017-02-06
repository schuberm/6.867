import numpy as np
import numpy.matlib
from plotBoundary import *
import pylab as pl
from itertools import compress
from scipy.spatial.distance import cdist,pdist,squareform
from utils import *
# import your LR training code

# load data from csv files
# train = loadtxt('data/data3_train.csv')
# X = train[:,0:2]
# Y = train[:,2:3]

# # Carry out training.
# epochs = 1000;
# lmbda = .02;
# gamma = 1e0;

# ### TODO: Compute the kernel matrix ###
#print(squareform(pdist(X,'euclidean')).shape)
#K = rbf(gamma,X)
### TODO: Implement train_gaussianSVM ###
#alphai, Xi, Yi, error = train_kPegasosSVM(X, Y, K, lmbda, epochs)

#(error)
# Define the predictSVM(x) function, which uses trained parameters
### TODO ###
def constructPredictor(w,Xi,Yi):
	#n,m = Xi.shape
	#K = zeros(n)
	def predict(x):
		#for i in range(n):
		# 	K[i] = np.exp(-gamma*np.linalg.norm(Xi[i]-x)**2)
		K = rbfx(gamma,Xi,x)
		return sum(np.squeeze(w)*np.squeeze(K))
	return predict

# Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
### TODO:  define predict_gaussianSVM(x) ###
# predict_gaussianSVM = constructPredictor(alphai,Xi,Yi)

# plot training results
#print '======Plot======'
#plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
#print '======Plot Show======'
#pl.show()
#pl.plot(error)
#pl.show()

if __name__=='__main__':
	# parameters
	Name = ['1','2','3','4']
	Gamma = [2e-2, 2e-1, 2e0, 2e1, 2e2];
	Lmbda = [0.02]

	# Name = ['4']
	# Gamma = [2e0];
	# Lmbda = [0.02]
	epochs = 1000;

	EM = np.zeros([len(Name),len(Gamma),len(Lmbda),2])
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


				margin[ni,gj,lk,0] = 1/np.linalg.norm(alphai)
				NSV[ni,gj,lk,0] = alphai.shape[0]
				predict_gaussianSVM = constructPredictor(alphai,Xi,Yi)

				corr = 0.0
				for i in range(n):
					corr = corr + np.abs(Y[i]-np.sign(predict_gaussianSVM (X[i,:])))
				print(corr/2/X.shape[0])
				EM[ni,gj,lk,0] = corr/2/X.shape[0]
				
				#plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
				#pl.show()
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
				EM[ni,gj,lk,1] = corr/2/X.shape[0]

	#print(EM)
	np.save('EM_gaussian_p.txt',EM)
	np.save('margin_gaussian_p.txt',margin)
	np.save('NSV_gaussian_p.txt',NSV)
