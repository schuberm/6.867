import numpy as np
from plotBoundary import *
import pylab as pl
import gradient_homebrew as gh
import scipy.optimize
import time
from utils import * #my functions for HW2
# import your LR training code

# parameters
name = '1'
print '======Training======'
# load data from csv files
train = np.loadtxt('data/data'+name+'_train.csv')
train1 = np.loadtxt('data/mnist_digit_3.csv')
train7 = np.loadtxt('data/mnist_digit_5.csv')

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

lmbda = 0.1
w = 0.1*np.random.rand(X.shape[1]+1)

batchNLL = makeNLL(X,Y,lmbda)
#print(batchNLL)
#wnew = gh.numeric_grad_descent(batchNLL,w)
wnew, tmp = scipy.optimize.fmin_bfgs(batchNLL,w,retall=True)

# Define the predictLR(x) function, which uses trained parameters
### TODO ###
@timefunc
def constructPredictor(w,b):
	def predict(x):
		print(x.shape)
		return np.sign(w.dot(x.T) + b)
	return predict

predictLR = constructPredictor(wnew[:w.shape[0]-1],wnew[-1])


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

YvLR = predictLR(Xv)

print(sum(Yv+YvLR))
print(Yv)
print(YvLR)

for i in range(2*vn):
	corr = corr + np.abs(Yv[i]-np.sign(predictLR(Xv[i,:])))
	if np.abs(Yv[i]-np.sign(predictLR(Xv[i,:]))) > 0.0001:
		dimg = np.reshape(Xv[i],[28,28])
		pl.imshow(dimg, cmap='Greys')
		pl.savefig('./figures/nist'+str(i)+'.pdf')
		pl.close()

#print(corr/2/X.shape[0])


# plot validation results
# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
# pl.show()
