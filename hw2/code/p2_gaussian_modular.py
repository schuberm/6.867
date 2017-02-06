import numpy as np
import numpy.matlib
from plotBoundary import *
import pylab as pl
import cvxopt
import cvxopt.solvers
from itertools import compress
#from scipy.spatial.distance import cdist
from utils import *
# import your SVM training code

# Carry out training, primal and/or dual
### TODO ###
def SVMtrain(X,Y,K,C):

	n,m = K.shape
	P = cvxopt.matrix(np.outer(Y,Y) * K)
	q = cvxopt.matrix(np.ones(n) * -1)
	A = cvxopt.matrix(Y, (1,n))
	b = cvxopt.matrix(0.0)

	tmp1 = np.diag(np.ones(n) * -1)
	tmp2 = np.identity(n)
	G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
	tmp1 = np.zeros(n)
	tmp2 = np.ones(n) * C
	h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

	tol = 1e-6
	solution = cvxopt.solvers.qp(P, q, G, h, A, b)
	alpha = np.array(solution['x'])
	#print(alpha.shape)
	#print(X.shape)
	alpha_1 = np.abs(alpha) > tol
	#print(alpha[alpha_1])
	indl = list(compress(xrange(len(alpha_1)), alpha_1))
	w = (Y*alpha).T.dot(X)

	#MASK
	Xi = X[indl,:]
	Yi = Y[indl]
	alphai = alpha[indl]
	#print(alphai)
	Ki = zeros(len(indl));
	#for i in range(len(indl)):
	#	Ki[i] = np.exp(-gamma*np.linalg.norm(Xi[i]-X[indl[0]])**2)
	#b = 1*Y[indl[0]]- sum(np.squeeze(alphai)*Ki)
	b = 0

	return alphai, Xi, Yi, b, alpha

# Define the predictSVM(x) function, which uses trained parameters
### TODO ###

def constructPredictor(w,gamma,Xi,Yi,b):
	#n,m = Xi.shape
	#K = zeros(n);
	def predict(x):
		#for i in range(n):
		#	K[i] = np.exp(-gamma*np.linalg.norm(Xi[i]-x)**2)
		#K = np.exp(-gamma*cdist(Xi,np.matlib.repmat(x, Xi.shape[0],1))**2)
		K = rbfx(gamma,Xi,x)
		return sum(np.squeeze(w)*np.squeeze(Yi)*np.squeeze(K)) #+ b
	return predict


if __name__=='__main__':
	# parameters
	Name = ['1','2','3','4']
	Gamma = [2e-2, 2e-1, 2e0, 2e1, 2e2];
	C = [0.01, 0.1, 1, 10, 100]

	# load data from csv files

	# Name = ['4']
	# Gamma = [2e0];
	# C = [1.0]

	#train = np.loadtxt('data/data1_train.csv')
	#X1 = train[:, 0:2].copy()

	#Construct kernel
	# n,m = X.shape
	# K = np.zeros((n, n))
	# for i in range(n):
	#     for j in range(n):
	#         K[i,j] = np.exp(-gamma*np.linalg.norm(X[i]-X[j])**2)

	#K1 = rbf(gamma,X1)


	EM = np.zeros([len(Name),len(Gamma),len(C),3])
	margin = np.zeros([len(Name),len(Gamma),len(C),1])
	NSV = np.zeros([len(Name),len(Gamma),len(C),1])
	for ni,name in enumerate(Name):
		for gj,gamma in enumerate(Gamma):
			for ck,c in enumerate(C):
				print '======Training======'
				train = np.loadtxt('data/data'+name+'_train.csv')
				# use deep copy here to make cvxopt happy
				X = train[:, 0:2].copy()
				Y = train[:, 2:3].copy()

				#Construct kernel
				# n,m = X.shape
				# K = np.zeros((n, n))
				# for i in range(n):
				#     for j in range(n):
				#         K[i,j] = np.exp(-gamma*np.linalg.norm(X[i,:]-X[j,:])**2)

				K = rbf(gamma,X)
				n,m = X.shape

				alphai,Xi,Yi,b,alpha = SVMtrain(X,Y,K,c)

				#ll = np.linalg.cholesky(K)

				margin[ni,gj,ck,0] = 1/np.linalg.norm(alpha.T.dot(K))
				NSV[ni,gj,ck,0] = alphai.shape[0]
				predictSVM = constructPredictor(alphai,gamma,Xi,Yi,b)

				# corr = 0.0
				# for i in range(n):
				# 	corr = corr + np.abs(Y[i]-np.sign(predictSVM(X[i,:])))
				# print(corr/2/X.shape[0])
				# EM[ni,gj,ck,0] = corr/2/X.shape[0]
				# # plot training results
				# #plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'Gaussian Kernel SVM Train')
				# #pl.savefig('./figures/ps23_train'+str(name)+'_'+str(gamma)+'_'+str(c)+'.pdf')
				# #pl.close()


				# print '======Validation======'
				# # load data from csv files
				# validate = np.loadtxt('./data/data'+name+'_validate.csv')
				# X = validate[:, 0:2]
				# Y = validate[:, 2:3]
				# # plot validation results
				# n, m = X.shape
				# corr = 0.0
				# for i in range(n):
				# 	corr = corr + np.abs(Y[i]-np.sign(predictSVM(X[i,:])))
				# print(corr/2/X.shape[0])
				# EM[ni,gj,ck,1] = corr/2/X.shape[0]
				# #plotDecisionBoundary(X, Y, predictSVM, [-1,0,1], title = 'Gaussian Kernel SVM Validate')
				# #pl.show()
				# #pl.savefig('./figures/ps23_validate'+str(name)+'_'+str(gamma)+'_'+str(c)+'.pdf')
				# #pl.close()
				# print '======Testing======'
				# # load data from csv files
				# validate = np.loadtxt('./data/data'+name+'_test.csv')
				# X = validate[:, 0:2]
				# Y = validate[:, 2:3]
				# # plot validation results
				# n, m = X.shape
				# corr = 0.0
				# for i in range(n):
				# 	corr = corr + np.abs(Y[i]-np.sign(predictSVM(X[i,:])))
				# print(corr/2/X.shape[0])
				# EM[ni,gj,ck,2] = corr/2/X.shape[0]
				#plotDecisionBoundary(X, Y, predictSVM, [-1,0,1], title = 'Gaussian Kernel SVM Test')
				#pl.show()
				#pl.savefig('./figures/ps23_validate'+str(name)+'_'+str(gamma)+'_'+str(c)+'.pdf')
				#pl.close()
		#plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
		#pl.show()

	# print(EM)
	#np.save('EM_gaussian.txt',EM)
	np.save('margin_gaussian.txt',margin)
	#np.save('NSV_gaussian.txt',NSV)