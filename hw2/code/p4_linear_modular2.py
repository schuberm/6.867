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
	Name = [('1','7'),('3','5'),('4','9'),('0','2','4','6','8')]

	# parameters
	#Name = [('1','7'),('3','5'),('4','9')]
	Gamma = [1];
	C = [0.01, 0.1, 1, 10, 100]

	#Gamma = [2e0];
	#C = [1]
	# load data from csv files

	# Name = ['4']
	# Gamma = [2e0];
	# C = [1.0]


	EM = np.zeros([len(Name),len(C),3])
	margin = np.zeros([len(Name),len(C),1])
	NSV = np.zeros([len(Name),len(C),1])
	for ni,name in enumerate(Name):
		for gj,gamma in enumerate(Gamma):
			for ck,c in enumerate(C):
				print '======Training======'
				if (ni < 3):
					train1 = np.loadtxt('data/mnist_digit_'+name[0]+'.csv')
					train7 = np.loadtxt('data/mnist_digit_'+name[1]+'.csv')

					n = 200
					X1 = train1[:n,:]
					X7 = train7[:n,:]
					X = np.vstack((X1,X7))
					#normalize
					X = 2*X/255-1
					print(X.shape)
					tmp1 = np.ones(n)
					tmp2 = np.ones(n)*-1
					Y = np.hstack((tmp1,tmp2))
					print(Y.shape)

				else:
					train0 = np.loadtxt('data/mnist_digit_0.csv')
					train1 = np.loadtxt('data/mnist_digit_1.csv')
					train2 = np.loadtxt('data/mnist_digit_2.csv')
					train3 = np.loadtxt('data/mnist_digit_3.csv')
					train4 = np.loadtxt('data/mnist_digit_4.csv')
					train5 = np.loadtxt('data/mnist_digit_5.csv')
					train6 = np.loadtxt('data/mnist_digit_6.csv')
					train7 = np.loadtxt('data/mnist_digit_7.csv')
					train8 = np.loadtxt('data/mnist_digit_8.csv')
					train9 = np.loadtxt('data/mnist_digit_9.csv')
					n = 200
					X0 = train0[:n,:]
					X1 = train1[:n,:]
					X2 = train2[:n,:]
					X3 = train3[:n,:]
					X4 = train4[:n,:]
					X5 = train5[:n,:]
					X6 = train6[:n,:]
					X7 = train7[:n,:]
					X8 = train8[:n,:]
					X9 = train9[:n,:]
					X = np.vstack((X0,X2,X4,X6,X8,X1,X3,X5,X7,X9))
					#normalize
					X = 2*X/255-1
					print(X.shape)
					tmp1 = np.ones(5*n)
					tmp2 = np.ones(5*n)*-1
					Y = np.hstack((tmp1,tmp2))
					print(Y.shape)

				n,m = X.shape
				K = np.zeros((n, n))
				for i in range(n):
					for j in range(n):
						K[i,j] = X[i].T.dot(X[j])
			        #K[i,j] = self.kernel(X[i], X[j])


				alphai,b,nsv = train_lQPSVM(X,Y,K,c)
				predictSVM = constructBPredictor(alphai,b)

				margin[ni,ck,0] = 1/np.linalg.norm(alphai)
				NSV[ni,ck,0] = nsv

				#print(alphai.shape)
				#print(predictSVM(X[0,:]).shape)
				corr = 0.0
				for i in range(n):
					corr = corr + np.abs(Y[i]-np.sign(predictSVM(X[i,:])))
				#corr = sum(np.abs(Y-np.sign(predictSVM(X[0,:]))))
				print(corr/2/X.shape[0])
				EM[ni,ck,0] = corr/2/X.shape[0]

				print '======Validation======'
				if (ni < 3):
					vn = 150
					X1v = train1[n:n+vn,:]
					X7v = train7[n:n+vn,:]
					Xv = np.vstack((X1v,X7v))
					#normalize
					Xv = 2*Xv/255-1
					tmp1 = np.ones(vn)	
					tmp2 = np.ones(vn)*-1
					Yv = np.hstack((tmp1,tmp2))
				else:
					vn = 150
					X0v = train0[n:n+vn,:]
					X1v = train1[n:n+vn,:]
					X2v = train2[n:n+vn,:]
					X3v = train3[n:n+vn,:]
					X4v = train4[n:n+vn,:]
					X5v = train5[n:n+vn,:]
					X6v = train6[n:n+vn,:]
					X7v = train7[n:n+vn,:]
					X8v = train8[n:n+vn,:]
					X9v = train9[n:n+vn,:]
					Xv = np.vstack((X0v,X2v,X4v,X6v,X8v,X1v,X3v,X5v,X7v,X9v))
					#normalize
					Xv = 2*Xv/255-1
					tmp1 = np.ones(5*vn)
					tmp2 = np.ones(5*vn)*-1
					Yv = np.hstack((tmp1,tmp2))

				n, m = Xv.shape
				corr = 0.0
				for i in range(n):
				 	corr = corr + np.abs(Yv[i]-np.sign(predictSVM(Xv[i,:])))
				print(corr/2/Xv.shape[0])
				#corr = sum(np.abs(Yv-np.sign(predictSVM(Xv))))
				EM[ni,ck,1] = corr/2/X.shape[0]

	np.save('EM_linear_4.txt',EM)
	np.save('margin_linear_4.txt',margin)
	np.save('NSV_linear_4.txt',NSV)
