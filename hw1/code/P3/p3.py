import numpy as np
import math
import gradient_check as gc
import regressData as lf
import matplotlib.pyplot as plt
import itertools
from sklearn import linear_model

#(X,y) = lf.regressBData()
(Xa,ya) = lf.regressAData()
(Xb,yb) = lf.regressBData()
(Xv,yv) = lf.validateData()
N = np.array(range(2,10))
#print(N)
#L = np.logspace(-8, -2, num=20)
L = np.linspace(1e-12, 1e-5, num=20)
sse = np.zeros((N.shape[0],L.shape[0]))

def rigde_regression_exact(N,l,X,y):
	#(X,y) = lf.regressAData()
	Phi = np.zeros((X.shape[0],N))
	for ii in range(X.shape[0]):
		Phi[ii,:]=[X[ii]**i for i in range(N)]
	w_reg = np.linalg.inv(l*np.eye(Phi.shape[1])+Phi.T.dot(Phi)).dot(Phi.T.dot(y))
	return w_reg

def rigde_regression_exact_sse(N,l,X,y):
	#(X,y) = lf.regressAData()
	Phi = np.zeros((X.shape[0],N))
	for ii in range(X.shape[0]):
		Phi[ii,:]=[X[ii]**i for i in range(N)]
	w_reg = np.linalg.inv(l*np.eye(Phi.shape[1])+Phi.T.dot(Phi)).dot(Phi.T.dot(y))
	sse = np.linalg.norm(w_reg.T.dot(Phi.T)-y)**2
	return sse

 # for n in range(N.shape[0]):
	# for l in range(L.shape[0]):
	# 	sse[n,l] = rigde_regression_exact_sse(N[n],L[l],X,y)

N = 4
#l = 0.0001
for l in np.logspace(-6, 2, num=5):
##Training on A
	w_reg = rigde_regression_exact(N,l,Xa,ya)
	Phi = np.zeros((Xa.shape[0],N))
	f = np.zeros((Xa.shape[0]))
	for ii in range(Xa.shape[0]):
		Phi[ii,:]=[Xa[ii]**i for i in range(N)]
		f[ii] = w_reg.T.dot(Phi[ii].T)


	pl2 = plt.plot(Xa,f,'s',label='lambda ='+str(l))
#plt.imshow(sse, cmap='hot', interpolation='nearest')
#plt.colorbar()
#plt.show()
pl1 = plt.plot(Xa,ya,'o',label='Training A')
pl3 = plt.plot(Xv,yv,'o',label='Validation')
pl4 = plt.plot(Xb,yb,'o',label='Testing B')
plt.legend(numpoints=1,loc='lower right')
#plt.savefig('../../figures/ps3_2_A_'+str(N)+'.pdf')
plt.clf()

for l in np.logspace(-6, 2, num=5):
##Training on A
	w_reg = rigde_regression_exact(N,l,Xb,yb)
	Phi = np.zeros((Xb.shape[0],N))
	f = np.zeros((Xb.shape[0]))
	for ii in range(Xb.shape[0]):
		Phi[ii,:]=[Xb[ii]**i for i in range(N)]
		f[ii] = w_reg.T.dot(Phi[ii].T)


	pl2 = plt.plot(Xb,f,'s',label='lambda ='+str(l))
#plt.imshow(sse, cmap='hot', interpolation='nearest')
#plt.colorbar()
#plt.show()
pl1 = plt.plot(Xa,ya,'o',label='Testing A')
pl3 = plt.plot(Xv,yv,'o',label='Validation')
pl4 = plt.plot(Xb,yb,'o',label='Training B')
plt.legend(numpoints=1,loc='lower right')
plt.savefig('../../figures/ps3_2_B_'+str(N)+'.pdf')
plt.clf()

#clf = linear_model.Ridge(alpha = 0.1)
#clf.fit(Phi,yb)
#print(clf.residues_)

