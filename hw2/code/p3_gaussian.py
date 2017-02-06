import numpy as np
import numpy.matlib
from plotBoundary import *
import pylab as pl
from itertools import compress
from scipy.spatial.distance import cdist,pdist,squareform
from utils import *
# import your LR training code

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
epochs = 1000;
lmbda = .02;
gamma = 1e0;

# n,m = X.shape
# K = zeros((n,n));
# ### TODO: Compute the kernel matrix ###
# for i in range(n):
#     for j in range(n):
#         K[i,j] = np.exp(-gamma*np.linalg.norm(X[i]-X[j])**2)

print(squareform(pdist(X,'euclidean')).shape)
K = rbf(gamma,X)
### TODO: Implement train_gaussianSVM ###

# def khingeLoss(X,Y,w,lmbda):
# 	n,m = X.shape
# 	tmp = 1-Y*(X.dot(w))
# 	stmp = sum(tmp > 0.0)
# 	return lmbda*np.linalg.norm(w)/2 + 1/n*stmp

# def kPegasos(X,Y,K,lmbda,epochs):
# 	n,m = X.shape
# 	t = 0
# 	eta = 0.1
# 	wt = zeros(n)
# 	epoch = 0
# 	max_epoch = epochs
# 	while (epoch < max_epoch):
# 		for i in range(n):
# 			t = t + 1
# 			eta = 1/(t*lmbda)
# 			cond = Y[i]*(wt.dot(K[:,i]))
# 			if (cond < 1):
# 				wt[i] = (1-eta*lmbda)*wt[i] + eta*Y[i]
# 				#print(wt)
# 			else:
# 				wt[i] = (1-eta*lmbda)*wt[i]
# 				#print(wt)
# 		epoch = epoch + 1
# 	return wt

# def train_gaussianSVM(X, Y, K, lmbda, epochs):

# 	alpha = kPegasos(X, Y, K, lmbda, epochs)

# 	print(alpha)

# 	#MASK
# 	tol = 1e-6
# 	alpha_1 = np.abs(alpha) > tol
# 	print(alpha[alpha_1])
# 	indl = list(compress(xrange(len(alpha_1)), alpha_1))
# 	Xi = X[indl,:]
# 	Yi = Y[indl]
# 	alphai = alpha[indl]
# 	return alphai, Xi, Yi

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
alphai, Xi, Yi = train_kPegasosSVM(X, Y, K, lmbda, epochs)
predict_gaussianSVM = constructPredictor(alphai,Xi,Yi)

# plot training results
print '======Plot======'
plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
print '======Plot Show======'
pl.show()
