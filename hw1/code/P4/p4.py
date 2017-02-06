import numpy as np
import math
import lassoData as lf
import matplotlib.pyplot as plt
import itertools
import sys
import pylab as pl
sys.path.insert(0, '/Users/mullspace/Dropbox (MIT)/currentcourses/6.867/hw1/code/p2')

import gradient_check as gc
from sklearn import linear_model

def fphi(X, y, w):
	#(X,y) = lf.lassoTrainData()
	n = X.shape[0]
	M = w.shape[0]
	Phi = np.zeros((n,M))
	for ii in range(n):
		Phi[ii,:]=[math.sin(0.4*X[ii]*i*np.pi) for i in range(M)]
	Phi[:,1] = X[ii]
	return Phi

def lasso(X, y, w, L):
	#(X,y) = lf.lassoTrainData()
	n = X.shape[0]
	M = w.shape[0]
	# Phi = np.zeros((n,M))
	# for ii in range(n):
	# 	Phi[ii,:]=[math.sin(0.4*X[ii]*i) for i in range(M)]
	# Phi[:,1] = X[ii]
	Phi = fphi(X, y, w)
	f = 1/n*np.sum((y-w.T.dot(Phi.T))**2)+L*np.sum(abs(w))
	return f

def rigde_regression_exact(X, y, l):
	#(X,y) = lf.lassoTrainData()
	n = X.shape[0]
	M = 13
	Phi = np.zeros((n,M))
	for ii in range(n):
		Phi[ii,:]=[math.sin(0.4*X[ii]*i*np.pi) for i in range(M)]
	Phi[:,1] = X[ii]
	w_reg = np.linalg.inv(l*np.eye(Phi.shape[1])+Phi.T.dot(Phi)).dot(Phi.T.dot(y))
	return w_reg

# def rigde_regression_exact_sse(N,l):
# 	(X,y) = lf.regressAData()
# 	Phi = np.zeros((X.shape[0],N))
# 	for ii in range(X.shape[0]):
# 		Phi[ii,:]=[X[ii]**i for i in range(N)]
# 	w_reg = np.linalg.inv(l*np.eye(Phi.shape[1])+Phi.T.dot(Phi)).dot(Phi.T.dot(y))
# 	sse = np.linalg.norm(w_reg.T.dot(Phi.T)-y)
# 	return sse

# for n in range(N.shape[0]):
# 	for l in range(L.shape[0]):
# 		sse[n,l] = rigde_regression_exact_sse(N[n],L[l])

w_true = pl.loadtxt('lasso_true_w.txt')
#w0 =np.random.randn(w_true.shape[0])
w0 =np.random.randn(13)

#w_gc = gc.numeric_grad_descent(lasso,w0,0.0000001)
#w_reg = rigde_regression_exact(X)
#print(w_reg.shape)
#print(lasso(w_reg,0.0000001))

(Xt,yt) = lf.lassoTrainData()
(Xv,yv) = lf.lassoValData()
(Xtt,ytt) = lf.lassoTestData()

##LASSO
clf = linear_model.Lasso(alpha = 0.1)
clf.fit(fphi(Xt,yt,w0),yt)
width = 1/1.5
plt.bar(range(len(clf.coef_)), clf.coef_, width, color="blue")
plt.title('LASSO')
#plt.savefig('../../figures/ps4_LASSO.pdf')
plt.clf()
##RR
l=1e-1;
w_reg = rigde_regression_exact(Xv,yv,l)
plt.bar(range(len(w_reg)), w_reg, width, color="blue")
plt.title('Ridge regression')
#plt.savefig('../../figures/ps4_RR_'+str(l)+'.pdf')
plt.clf()
##RRscikit
clrr = linear_model.Ridge (alpha = l)
clrr.fit(fphi(Xt,yt,w0),yt)
print(clrr.coef_[0])
plt.bar(range(len(clrr.coef_[0])), clrr.coef_[0], width, color="blue")
plt.title('Ridge regression scikit')
plt.savefig('../../figures/ps4_RRsk_'+str(l)+'.pdf')
plt.clf()

x = np.linspace(-1,1,50)
#f = w_true.dot(fphi(w_true).T)
#f2 = w_reg.T.dot(fphi(w_reg).T)
plt.plot(Xt,yt,'o',label='Training')
plt.plot(Xv,yv,'s',label='Validation')
plt.plot(Xtt,ytt,'x',label='Testing')
plt.plot(x,w_true.dot(fphi(x,yt,w_true).T),label='True')
plt.plot(x,(w_reg.T.dot(fphi(x,yt,w_reg).T)).T,label='Ridge lambda=1')
plt.plot(x,clf.coef_.dot(fphi(x,yt,clf.coef_).T),label='LASSO lambda=0.1')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()
plt.legend(numpoints=1,loc='lower right')
#plt.savefig('../../figures/ps4.pdf')
plt.clf()

