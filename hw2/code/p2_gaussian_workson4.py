import numpy as np
import numpy.matlib
from plotBoundary import *
import pylab as pl
import cvxopt
import cvxopt.solvers
from itertools import compress
from scipy.spatial.distance import cdist
# import your SVM training code

# parameters
name = '4'
gamma = 2e-1;
print '======Training======'
# load data from csv files
train = np.loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

# Carry out training, primal and/or dual
### TODO ###
n,m = X.shape
K = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        #K[i,j] = self.kernel(X[i], X[j])
        #K[i,j] = X[i].T.dot(X[j])
        K[i,j] = np.exp(-gamma*np.linalg.norm(X[i]-X[j])**2)

P = cvxopt.matrix(np.outer(Y,Y) * K)
#print(np.linalg.eigvals(P))
q = cvxopt.matrix(np.ones(n) * -1)
A = cvxopt.matrix(Y, (1,n))
b = cvxopt.matrix(0.0)
#G = cvxopt.matrix(np.diag(np.ones(n) * -1))
#h = cvxopt.matrix(np.zeros(n))

C = 1.0
tmp1 = np.diag(np.ones(n) * -1)
tmp2 = np.identity(n)
G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
tmp1 = np.zeros(n)
tmp2 = np.ones(n) * C
h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

tol = 1e-6
solution = cvxopt.solvers.qp(P, q, G, h, A, b)
alpha = np.array(solution['x'])
print(alpha.shape)
print(X.shape)
alpha_1 = np.abs(alpha) > tol
print(alpha[alpha_1])
indl = list(compress(xrange(len(alpha_1)), alpha_1))
w = (Y*alpha).T.dot(X)
#print(w)
#print(X[alpha_1])
# for i in range(n):
# 	K[i] = np.exp(-gamma*np.linalg.norm(X[i]-X[indl[0]])**2)
# b = 1*Y[indl[0]]- sum(np.squeeze(alpha)*np.squeeze(Y)*K)
# print(b)

#MASK
Xi = X[indl,:]
Yi = Y[indl]
alphai = alpha[indl]
print(alphai)
Ki = zeros(len(indl));
for i in range(len(indl)):
	Ki[i] = np.exp(-gamma*np.linalg.norm(Xi[i]-X[indl[0]])**2)
b = 1*Y[indl[0]]- sum(np.squeeze(alphai)*Ki)

# Define the predictSVM(x) function, which uses trained parameters
### TODO ###

def constructPredictor(w,Xi,Yi,b):
	n,m = Xi.shape
	K = zeros(n);
	def predict(x):
		for i in range(n):
			K[i] = np.exp(-gamma*np.linalg.norm(Xi[i]-x)**2)
		#K = np.exp(-gamma*cdist(Xi,np.matlib.repmat(x, Xi.shape[0],1))**2)
		return sum(np.squeeze(w)*np.squeeze(Yi)*K) #+ b
	return predict

predictSVM = constructPredictor(alphai,Xi,Yi,b)


# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


print '======Validation======'
# load data from csv files
validate = np.loadtxt('./data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()
