import numpy as np
from plotBoundary import *
import pylab as pl
from utils import *
# import your LR training code


# load data from csv files
# train = loadtxt('data/data4_train.csv')
# X = train[:,0:2]
# Y = train[:,2:3]

# Carry out training.
### TODO %%%
def hingeLoss(X,Y,w,lmbda):
    n,m = X.shape
    tmp = 1-np.squeeze(Y)*np.squeeze(X.dot(w))
    #print(np.squeeze(Y).shape)
    #print(np.squeeze(X.dot(w)).shape)
    #print(sum(tmp[tmp > 0.0]))
    stmp = sum(tmp[tmp > 0.0])
    return lmbda*np.linalg.norm(w)/2 + 1/n*stmp


# def lPegasos (X,Y,lmbda,max_epoch):
# 	n,m = X.shape
# 	t = 0
# 	eta = 0.1
# 	#lmbda = 0.002
# 	#wt = np.array([0.0, 0.0, 0.0])
# 	wt = np.zeros(X.shape[1]+1)
# 	epoch = 0
# 	#max_epoch = 
# 	error = []
# 	while (epoch < max_epoch):
# 		for i in range(n):
# 			t = t + 1
# 			eta = 1/(t*lmbda)
# 			if (Y[i]*(wt[1:].dot(X[i,:])+wt[0]) < 1):
# 				wt[0] =  wt[0] + eta*Y[i]
# 				wt[1:] = (1-eta*lmbda)*wt[1:] + eta*Y[i]*X[i,:]
# 				#print(wt)
# 			else:
# 				#wt[0] =  wt[0] + eta*Y[i]
# 				#wt[1:]  = (1-eta*lmbda)*wt[1:]
# 				wt  = (1-eta*lmbda)*wt
# 				#print(wt)
# 		epoch = epoch + 1
# 		error.append(hingeLoss(X,Y,wt[1:],lmbda))
# 	return wt, error

# Define the predict_linearSVM(x) function, which uses global trained parameters, w
### TODO: define predict_linearSVM(x) ###

def constructPredictor(w):
	def predict(x):
		return w[1:].dot(x)+w[0]
	return predict

#print(wt)
# max_epoch = 100
# lmbda = 0.2
# wt, error = lPegasos(X,Y,lmbda,max_epoch)
# predict_linearSVM = constructPredictor(wt)

# # plot training results
# print '======Plot======'
# #plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
# #pl.show()

# pl.plot(error)
# pl.show()

if __name__=='__main__':
	# parameters
	Name = ['1','2','3','4']
	Gamma = [2];
	Lmbda = [2, 2e-1, 2e-2, 2e-3, 2e-4, 2e-5, 2e-6, 2e-7, 2e-8, 2e-9, 2e-10]

	# Name = ['4']
	# Gamma = [2e0];
	# Lmbda = [0.02]
	max_epoch = 100;

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

				#K = rbf(gamma,X)
				### TODO: Implement train_gaussianSVM ###
				alphai, error, nsv = lPegasos(X,Y,lmbda,max_epoch)


				margin[ni,gj,lk,0] = 1/np.linalg.norm(alphai)
				NSV[ni,gj,lk,0] = nsv
				predict_linearSVM = constructPredictor(alphai)

				corr = 0.0
				for i in range(n):
					corr = corr + np.abs(Y[i]-np.sign(predict_linearSVM (X[i,:])))
				print(corr/2/X.shape[0])
				EM[ni,gj,lk,0] = corr/2/X.shape[0]
				
				plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear Kernel Pegasos SVM')
				#pl.show()
				pl.savefig('./figures/ps31_training'+str(name)+'_'+str(gamma)+'_'+str(lmbda)+'.pdf')
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
					corr = corr + np.abs(Y[i]-np.sign(predict_linearSVM (X[i,:])))
				print(corr/2/X.shape[0])

				plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear Kernel Pegasos SVM')
				#pl.show()
				pl.savefig('./figures/ps31_validate'+str(name)+'_'+str(gamma)+'_'+str(lmbda)+'.pdf')
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
					corr = corr + np.abs(Y[i]-np.sign(predict_linearSVM (X[i,:])))
				print(corr/2/X.shape[0])

				plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear Kernel Pegasos SVM')
				#pl.show()
				pl.savefig('./figures/ps31_test'+str(name)+'_'+str(gamma)+'_'+str(lmbda)+'.pdf')
				pl.close()
				EM[ni,gj,lk,2] = corr/2/X.shape[0]


	#print(EM)
	np.save('EM_linear_p.txt',EM)
	np.save('margin_linear_p.txt',margin)
	np.save('NSV_linear_p.txt',NSV)

