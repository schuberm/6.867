#from numpy import *
import numpy as np
from plotBoundary import *
import pylab as pl
import gradient_homebrew as gh
import scipy.optimize
from utils import * 
# import your LR training code

# parameters
name = '1'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
### TODO ###
# def makeNLL(x,y):
# 	def NLL(wn):
# 		n,m = x.shape
# 		w = wn[:m]
# 		b = wn[m]
# 		yp = (w.dot(x.T)+b)
# 		return sum(np.log(1+np.exp(yp*y[:,0]*-1)))
# 	return NLL
def yPredicted(w, b, phi):
    return sigmoid(phi.dot(w) + b)

def error(phi, y, lamduh):
	def NLL(wb):
	    n,m = phi.shape
	    w = wb[:m]
	    b = wb[m]
	    yp = yPredicted(w, b, phi)
	    if not yp.shape == (n,1):
	        yp = yp.reshape((n,1))
	    return sum(np.log(1 + np.exp(-yp*y))) + lamduh * (w.T.dot(w))
	return NLL

b = 10
w = np.array([0.1, 0.1, b])
lmbda = [0.0, 1.0]

for l in lmbda:
	batchNLL = error(X,Y,l)
	#wnew = gh.numeric_grad_descent(batchNLL,w)
	wnew, tmp = scipy.optimize.fmin_bfgs(batchNLL,w,retall=True)
	print(tmp)
	normlist = []
	for t in tmp:
		normlist.append(np.linalg.norm(t))

	pl.plot(normlist)
	pl.xlabel('Number of Iterations')
	pl.ylabel('Norm of weights')
	
pl.legend(['$\lambda$ = 0.0','$\lambda$ = 1.0'])
pl.savefig('./figures/ps1_1.pdf')

# Define the predictLR(x) function, which uses trained parameters
### TODO ###
def constructPredictor(w,b):
	def predict(x):
		return sigmoid(w.dot(x) + b)
	return predict

predictLR = constructPredictor(wnew[:wnew.shape[0]-1],wnew[-1])

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
pl.show()

print '======Test======'
# load data from csv files
validate = loadtxt('data/data'+name+'_test.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Test')
pl.show()
