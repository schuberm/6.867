import numpy as np
from plotBoundary import *
import pylab as pl
import cvxopt
import cvxopt.solvers
from itertools import compress
from utils import *
# import your SVM training code

if __name__=='__main__':
	# parameters
	Name = ['1','2','3','4']
	C = [0.01, 0.1, 1, 10, 100]

	# load data from csv files

	# Name = ['4']
	# Gamma = [2e0];
	# C = [1.0]

	EM = np.zeros([len(Name),len(C),3])
	margin = np.zeros([len(Name),len(C),1])
	NSV = np.zeros([len(Name),len(C),1])
	for ni,name in enumerate(Name):
		for ck,c in enumerate(C):
			print '======Training======'
			train = np.loadtxt('data/data'+name+'_train.csv')
			# use deep copy here to make cvxopt happy
			X = train[:, 0:2].copy()
			Y = train[:, 2:3].copy()
			n,m = X.shape
			K = np.zeros((n, n))
			for i in range(n):
			    for j in range(n):
			        #K[i,j] = self.kernel(X[i], X[j])
			        K[i,j] = X[i].T.dot(X[j])

			alphai,b,nsv = train_lQPSVM(X,Y,K,c)
			predictSVM = constructBPredictor(alphai,b)

			margin[ni,ck,0] = 1/np.linalg.norm(alphai)
			NSV[ni,ck,0] = nsv

			corr = 0.0
			for i in range(n):
				corr = corr + np.abs(Y[i]-np.sign(predictSVM(X[i,:])))
			print(corr/2/X.shape[0])
			EM[ni,ck,0] = corr/2/X.shape[0]
			# plot training results
			#plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')
			#pl.savefig('./figures/ps21_train'+str(name)+'_'+str(c)+'.pdf')
			#pl.close()


			print '======Validation======'
			# load data from csv files
			validate = np.loadtxt('./data/data'+name+'_validate.csv')
			X = validate[:, 0:2]
			Y = validate[:, 2:3]
			# plot validation results
			n, m = X.shape
			corr = 0.0
			for i in range(n):
				corr = corr + np.abs(Y[i]-np.sign(predictSVM(X[i,:])))
			print(corr/2/X.shape[0])
			EM[ni,ck,1] = corr/2/X.shape[0]
			#plotDecisionBoundary(X, Y, predictSVM, [-1,0,1], title = 'SVM Validate')
			#pl.show()
			#pl.savefig('./figures/ps21_validate'+str(name)+'_'+str(c)+'.pdf')
			#pl.close()

			print '======Testing======'
			# load data from csv files
			validate = np.loadtxt('./data/data'+name+'_test.csv')
			X = validate[:, 0:2]
			Y = validate[:, 2:3]
			# plot validation results
			n, m = X.shape
			corr = 0.0
			for i in range(n):
				corr = corr + np.abs(Y[i]-np.sign(predictSVM(X[i,:])))
			print(corr/2/X.shape[0])
			EM[ni,ck,2] = corr/2/X.shape[0]
			#plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Test')
			#pl.savefig('./figures/ps21_test'+str(name)+'_'+str(c)+'.pdf')
			#pl.close()
		#pl.show()

	# print(EM)
	np.save('EM_linear.txt',EM)
	np.save('margin_linear.txt',margin)
	np.save('NSV_linear.txt',NSV)
