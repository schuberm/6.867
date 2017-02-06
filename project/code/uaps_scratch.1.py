import time
import numpy as np
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
import load_datasets as ld
import pickle
import numpy.matlib
#import pylab as plt
from scipy.optimize import minimize, fmin_cobyla
from random import randint
from joblib import Parallel, delayed
from time import sleep

def predictor (loss):
		def predict(x):
			return numpy.asscalar(numpy.argmax(loss(x), axis=1))
		return predict

def Err(X, v):
	m,n = X.shape
	ind = 0.0
	for ii in range(m):
		if nnpredict(X[ii,:]+v) != nnpredict(X[ii,:]):
			ind = 1.0 + ind
	return ind/m

def l2(r):
	return numpy.linalg.norm(r)

def constraints(k,x,v):
	def consr(r):
		print r
		return numpy.abs(k(x+v+r) - k(x))
	return consr

def constargs(r,*args):
	#return numpy.abs(float(nnpredict(args[0]+args[1]+r)) - float(nnpredict(args[1])))
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
		return numpy.linalg.norm(v - vp)
	return p

def vpconswrap(eta):
	def vpcons(vp):
		return eta - numpy.linalg.norm(vp)
	return vpcons

def uap(X,zeta=20000,delta=0.1,disp=False):
	m,n = X.shape
	eps = 0.1
	v = numpy.zeros([m,n])
	count = 0
	cons  = []
	#while Err(X,v) > 1-delta:
	while count < 1:
		for jj in range(m):
		#for jj in range(1):
			ii = jj
			if nnpredict(X[ii,:]+v[ii,:]) == nnpredict(X[ii,:]):
				#print ii
				r0 = numpy.random.randn(n)
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
					#r0 = randint(0,10)*numpy.random.randn(n)
					r0 = numpy.random.randn(n)
					result = minimize(l2, r0, method = 'COBYLA', constraints = cons, options={'disp': disp, 'catol': 0.1, 'rhobeg':50.0, 'tol':zeta})
					print result.status
					#print "c x"

				v[ii,:] = result.x
				#print v
				#print "compare after p"
				#print nnpredict(X[ii])
				#print nnpredict(X[ii,:]+v[ii,:])
		count = count + 1
	return v


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
	#print data0['y_train']

	# if rl2d== True:
	# 	data0 = ld.make2Ddata(name)
	# 	pickle.dump( data0, open( "1.p", "wb" ) )
	# else:
	# 	data0 = pickle.load( open("1.p", "rb" ) )

	dim = data0['X_train'].shape[1]

	model = FullyConnectedNet([100,100,100,100], input_dim=dim, num_classes=len(mnistlist),
	          weight_scale=weight_scale, dtype=numpy.float64)
	solver = Solver(model, data0,
	            print_every=80, num_epochs=20, batch_size=50,
	            update_rule='sgd',
	            optim_config={
	              'learning_rate': learning_rate,
	            }
	     )
	solver.train()


	global nnpredict 
	nnpredict = predictor(solver.model.loss)

	#X = data0['X_train']
	#print X[0,:]
	#m,n = X.shape
	#eps = 0.1
	#v = numpy.zeros(n)
	#v0 = 0*numpy.random.randn(n)
	#print nnpredict(X[0,:])
	#print nnpredict(X[0,:]+v0)
	#consr = constraints (nnpredict, X[0,:], v)
	#print consr(v0)
	#r0 = numpy.random.randn(n)

	#result = uap(data0['X_train'])
	#numpy.savetxt('v.txt', v)
	#Parallel(n_jobs=20, verbose=5)(delayed(sleep)(.1) for _ in range(10)) 
	np = 4
	ms = 16
	#results = Parallel(n_jobs=-1,backend="multiprocessing", verbose=5)((delayed(uap)( data0['X_train'])) for ii in range(np))
	results = Parallel(n_jobs=-1,backend="multiprocessing", verbose=5)((delayed(uap)( data0['X_train'][ii*int(ms/np):(ii+1)*int(ms/np),:])) for ii in range(np))
	#results = Parallel(n_jobs=-1,backend="multiprocessing")(map(delayed(uap), data0['X_train']))
	#print results[1].shape
	pickle.dump( results, open( "uap.p", "wb" ) )
	#numpy.savetxt('v.txt', results)

