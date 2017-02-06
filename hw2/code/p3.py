#from numpy import *
import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code


# load data from csv files
train = loadtxt('data/data4_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

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


def lPegasos (X,Y,lmbda,max_epoch):
	n,m = X.shape
	t = 0
	eta = 0.1
	#lmbda = 0.002
	#wt = np.array([0.0, 0.0, 0.0])
	wt = np.zeros(X.shape[1]+1)
	epoch = 0
	#max_epoch = 
	error = []
	while (epoch < max_epoch):
		for i in range(n):
			t = t + 1
			eta = 1/(t*lmbda)
			if (Y[i]*(wt[1:].dot(X[i,:])+wt[0]) < 1):
				wt[0] =  wt[0] + eta*Y[i]
				wt[1:] = (1-eta*lmbda)*wt[1:] + eta*Y[i]*X[i,:]
				#print(wt)
			else:
				#wt[0] =  wt[0] + eta*Y[i]
				#wt[1:]  = (1-eta*lmbda)*wt[1:]
				wt  = (1-eta*lmbda)*wt
				#print(wt)
		epoch = epoch + 1
		error.append(hingeLoss(X,Y,wt[1:],lmbda))
	return wt, error

# Define the predict_linearSVM(x) function, which uses global trained parameters, w
### TODO: define predict_linearSVM(x) ###

def constructPredictor(w):
	def predict(x):
		return w[1:].dot(x)+w[0]
	return predict

#print(wt)
max_epoch = 100
lmbda = 0.2
wt, error = lPegasos(X,Y,lmbda,max_epoch)
predict_linearSVM = constructPredictor(wt)

# plot training results
print '======Plot======'
#plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
#pl.show()

pl.plot(error)
pl.show()
