import time
import numpy as np
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
import load_datasets as ld
import pickle
import numpy.matlib
from scipy.optimize import minimize, fmin_cobyla
from random import randint
from joblib import Parallel, delayed

def uapcalc(uapobj,X,zeta=20000,delta=0.1,disp=False):

	uapobj.nnpredict = uapobj.predictor(uapobj.loss)

	def Err(X, v):
		m,n = X.shape
		ind = 0.0
		for ii in range(m):
			if uapobj.nnpredict(X[ii,:]+v) != uapobj.nnpredict(X[ii,:]):
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
		#return numpy.abs(float(uapobj.nnpredict(args[0]+args[1]+r)) - float(uapobj.nnpredict(args[1])))
		if uapobj.nnpredict(args[1]) == 0:
			return float(uapobj.nnpredict(args[0]+args[1]+r))-1.0
		else:
			return -float(uapobj.nnpredict(args[0]+args[1]+r))

	def linconst1(r,*args):
		eps = 0.2
		M = 1e10
		if uapobj.nnpredict(args[0]+args[1]+r) - uapobj.nnpredict(args[1]) > eps:
			y = 1
		else:
			y = 0
		return -(uapobj.nnpredict(args[0]+args[1]+r) - uapobj.nnpredict(args[1])) - eps + M*y 

	def linconst2(r,*args):
		eps = 0.2
		M = 1e10
		if uapobj.nnpredict(args[0]+args[1]+r) - uapobj.nnpredict(args[1]) > eps:
			y = 1
		else:
			y = 0
		return (uapobj.nnpredict(args[0]+args[1]+r) - uapobj.nnpredict(args[1])) - eps + M*(1-y) 

	def pp(v):
		def p(vp):
			return numpy.linalg.norm(v - vp)
		return p

	def vpconswrap(eta):
		def vpcons(vp):
			return eta - numpy.linalg.norm(vp)
		return vpcons

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
			if uapobj.nnpredict(X[ii,:]+v[ii,:]) == uapobj.nnpredict(X[ii,:]):
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
				#print uapobj.nnpredict(X[ii])
				#print uapobj.nnpredict(X[ii,:]+v[ii,:])
		count = count + 1
	return v

class uap:

	def __init__(self,loss):
		self.loss = loss

	def predictor(self,loss):
		def predict(x):
			return numpy.asscalar(numpy.argmax(loss(x), axis=1))
		return predict

	# def uap_parallel(self,data):
	# 	np = 4
	# 	ms = 16
	# 	results = Parallel(n_jobs=-1,backend="multiprocessing", verbose=5)((delayed(uapcalc)( self, data[ii*int(ms/np):(ii+1)*int(ms/np),:])) for ii in range(np))
	# 	return results