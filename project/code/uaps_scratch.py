import time
import numpy as np
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
import load_datasets as ld
import pickle
import numpy.matlib
import pylab as plt
from scipy.optimize import minimize, fmin_cobyla
from random import randint


if __name__=='__main__':

	#x = ld.loadMNIST('0')
	weight_scale = 0.1
	learning_rate = 1e-2
	rlMNIST = False
	rl2d = True
	name = '2'
	figurename = './figures/mnist.1l.5.0.01.pdf'

	mnistlist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	

	if rlMNIST == True:
		#data0 = ld.makeMNIST(['0','1'], 100, 50, 50)
		data0 = ld.makeMNIST(mnistlist, 500, 50, 50)
		pickle.dump( data0, open( "mnist.p", "wb" ) )
	else:
		data0 = pickle.load( open( "mnist.p", "rb" ) )

	dim = data0['X_train'].shape[1]
	print data0['y_train']

	# if rl2d== True:
	# 	data0 = ld.make2Ddata(name)
	# 	pickle.dump( data0, open( "1.p", "wb" ) )
	# else:
	# 	data0 = pickle.load( open("1.p", "rb" ) )

	dim = data0['X_train'].shape[1]

	model = FullyConnectedNet([100,100], input_dim=dim, num_classes=len(mnistlist),
	          weight_scale=weight_scale, dtype=np.float64)
	solver = Solver(model, data0,
	            print_every=80, num_epochs=20, batch_size=50,
	            update_rule='sgd',
	            optim_config={
	              'learning_rate': learning_rate,
	            }
	     )
	solver.train()

	def predictor (loss):
		def predict(x):
			return np.asscalar(np.argmax(loss(x), axis=1))
		return predict

	nnpredict = predictor(solver.model.loss)

	def Err(X, v):
		m,n = X.shape
		ind = 0.0
		for ii in range(m):
			if nnpredict(X[ii,:]+v) != nnpredict(X[ii,:]):
				ind = 1.0 + ind
		return ind/m

	def l2(r):
		return np.linalg.norm(r)

	def constraints(k,x,v):
		def consr(r):
			print r
			return np.abs(k(x+v+r) - k(x))
		return consr

	def constargs(r,*args):
		#return np.abs(float(nnpredict(args[0]+args[1]+r)) - float(nnpredict(args[1])))
		if nnpredict(args[1]) == 0:
			return float(nnpredict(args[0]+args[1]+r))-1.0
		else:
			return -float(nnpredict(args[0]+args[1]+r))

	def linconst1(r,*args):
		eps = 0.2
		M = 1e10
		if nnpredict(args[0]+args[1]+r) - nnpredict(args[1]) > eps:
			y = 1
		else:
			y = 0
		return -(nnpredict(args[0]+args[1]+r) - nnpredict(args[1])) - eps + M*y 

	def linconst2(r,*args):
		eps = 0.2
		M = 1e10
		if nnpredict(args[0]+args[1]+r) - nnpredict(args[1]) > eps:
			y = 1
		else:
			y = 0
		return (nnpredict(args[0]+args[1]+r) - nnpredict(args[1])) - eps + M*(1-y) 

	def pp(v):
		def p(vp):
			return np.linalg.norm(v - vp)
		return p

	def vpconswrap(eta):
		def vpcons(vp):
			return eta - np.linalg.norm(vp)
		return vpcons

	def uap(X,zeta=20000,delta=0.1,disp=False):
		m,n = X.shape
		eps = 0.1
		#v = 1.0*np.zeros(n)
		#v = np.random.randn(n)
		v = np.zeros([m,n])
		count = 0
		cons  = []
		#while Err(X,v) > 1-delta:
		while count < 1:
			for jj in range(m):
				ii = randint(0,m-1)
				if nnpredict(X[ii,:]+v[ii,:]) == nnpredict(X[ii,:]):
					print ii
					r0 = np.random.randn(n)
					#print nnpredict(X[ii,:])
					# consr = constraints (nnpredict, X[ii,:], v)
					# cons = ({'type': 'ineq', 'fun': consr})
					# result = minimize(l2, r0, method = 'COBYLA', constraints = cons, options={'disp': True, 'maxiter':100})
					# deltav = result.x
					# print consr(deltav)
					# #print deltav
					# print nnpredict(X[ii,:]+deltav)
					cons = ({'type': 'ineq', 'fun': linconst1, 'args':(v[ii,:],X[ii,:])},
					 		{'type': 'ineq', 'fun': linconst2, 'args':(v[ii,:],X[ii,:])})
					# c1 = {'type': 'ineq', 'fun': linconst1, 'args':(v,X[ii,:])}
					# c2 = {'type': 'ineq', 'fun': linconst2, 'args':(v,X[ii,:])}
					# cons.append(c1)
					# cons.append(c2)
					result = minimize(l2, r0, method = 'COBYLA', constraints = cons, options={'disp': disp, 'catol': 0.1, 'rhobeg':50.0, 'tol':zeta})
					
					print result.status
					#print "c 1"

					while result.success == False:
						#r0 = randint(0,10)*np.random.randn(n)
						r0 = np.random.randn(n)
						result = minimize(l2, r0, method = 'COBYLA', constraints = cons, options={'disp': disp, 'catol': 0.1, 'rhobeg':50.0, 'tol':zeta})
						print result.status
						#print "c x"

					#print result.status


					#print X[ii,:]
					# print "compare after c"
					# print nnpredict(X[ii,:])
					# print nnpredict(X[ii,:]+result.x+v)
					# deltav = result.x
					# pfun = pp(v+deltav)
					# vcons = vpconswrap(zeta)
					# v0 = np.random.randn(n)
					# #cons = ({'type': 'ineq', 'fun': vcons})
					# cons = ({'type': 'ineq', 'fun': linconst1, 'args':(v,X[ii,:])},
					# 		{'type': 'ineq', 'fun': linconst2, 'args':(v,X[ii,:])},
					# 		{'type': 'ineq', 'fun': vcons})
					# result = minimize(pfun, v, method = 'COBYLA', constraints = cons, options={'disp': disp, 'rhobeg':50.0, 'tol':zeta, 'maxiter': 1000})
					# #print "p 1"
					# #print result.status
					# while result.success == False:
					# 	v0 = np.random.randn(n)
					# 	result = minimize(pfun, v0, method = 'COBYLA', constraints = cons, options={'disp': disp, 'rhobeg':50.0, 'tol':zeta, 'maxiter': 1000})
					# 	#print "p x"
					# 	#print result.status

					v[ii,:] = result.x
					#print v
					#print "compare after p"
					print nnpredict(X[ii])
					print nnpredict(X[ii,:]+v[ii,:])
			count = count + 1
		return v


	X = data0['X_train']
	print X[0,:]
	m,n = X.shape
	eps = 0.1
	v = np.zeros(n)
	v0 = 0*np.random.randn(n)
	print nnpredict(X[0,:])
	print nnpredict(X[0,:]+v0)
	#consr = constraints (nnpredict, X[0,:], v)
	#print consr(v0)
	r0 = np.random.randn(n)
	#consr = constraints (nnpredict, X[0,:], v)
	#cons = ({'type': 'ineq', 'fun': lambda r: consr(r)})
	#result = minimize(l2, r0, method = 'COBYLA', constraints = cons, options={'disp': True, 'maxiter':100})
	#deltav = result.x
	#print consr(deltav)
	#print np.asscalar(np.abs(nnpredict(X[0,:]+v0+r0) - nnpredict(X[0,:])))
	#results = fmin_cobyla(l2,r0, cons=constargs, consargs=(v0,X[0,:]), disp = 2, rhobeg = 10.0, rhoend=50.0 , catol = 0.1)
	#results = fmin_cobyla(l2,r0, cons=[linconst1,linconst2], consargs=(v0,X[0,:]), disp = 2, rhobeg = 10.0, rhoend=50.0 , catol = 0.1)

	#print results


	###########THIS WORKS#####################################################
	# cons = ({'type': 'ineq', 'fun': linconst1, 'args':(v0,X[0,:])},
	# 		{'type': 'ineq', 'fun': linconst2, 'args':(v0,X[0,:])})
	# result = minimize(l2, r0, method = 'COBYLA', constraints = cons, options={'disp': True, 'catol': 0.1, 'rhobeg':10.0, 'tol':50.0})
	# print result

	# while result.success == False:
	# 	r0 = 10*np.random.randn(n)
	# 	result = minimize(l2, r0, method = 'COBYLA', constraints = cons, options={'disp': True, 'catol': 0.1, 'rhobeg':10.0, 'tol':0.1})
	# 	print result


	# print X[0,:]
	# print nnpredict(X[0,:])
	# print nnpredict(X[0,:]+result.x+v0)
	##########################################################################

	v = uap(data0['X_train'])
	#print v

	# count = 0
	# for ii in range(m):
	# 	print ii
	# 	print nnpredict(X[ii,:])
	# 	print nnpredict(X[ii,:]+v)
	# 	if nnpredict(X[ii,:]) == nnpredict(X[ii,:]+v):
	# 		count = count + 1

	# print count
