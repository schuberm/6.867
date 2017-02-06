import time
import numpy as np
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
import load_datasets as ld
import dill as pickle
import numpy.matlib
import pylab as plt
from scipy.optimize import minimize, fmin_cobyla
from sklearn.preprocessing import normalize
from random import randint
from joblib import Parallel, delayed
from time import sleep


def low_rank_approx(SVD=None, A=None, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    if not SVD:
        SVD = numpy.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = numpy.zeros((len(u), len(v)))
    for i in xrange(r):
        Ar += s[i] * numpy.outer(u.T[i], v[i])
    return Ar


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
				#print result.status
				#print "c 1"

				while result.success == False:
					#r0 = randint(0,10)*numpy.random.randn(n)
					r0 = numpy.random.randn(n)
					result = minimize(l2, r0, method = 'COBYLA', constraints = cons, options={'disp': disp, 'catol': 0.1, 'rhobeg':50.0, 'tol':zeta})
					#print result.status
					#print "c x"

				v[ii,:] = result.x
				#print v
				#print "compare after p"
				#print nnpredict(X[ii])
				#print nnpredict(X[ii,:]+v[ii,:])
		count = count + 1
	return v


if __name__=='__main__':


	#from scipy.misc import lena
	import pylab

	# weight_scale = 0.2
	# learning_rate = 0.1
	weight_scale = 0.1
	learning_rate = 1e-2
	filename = "mnist"
	train = True
	global nnpredict 

	if train == True:
		#data0 = ld.makeSpiralSet(N=100,D=50,K=3)
		#data0 = ld.makeRandSet(N=200,D=dim,K=nclasses)
		data0 = pickle.load( open( "mnist.p", "rb" ) )
		#pickle.dump( data0, open( "data."+filename+".p", "wb" ) )
		dim = data0['X_train'].shape[1]
		nclasses = data0['X_train'].shape[0]
		model = FullyConnectedNet([100,100,100,100], input_dim=dim, num_classes=nclasses,
		          weight_scale=weight_scale, dtype=numpy.float64, reg=0.00)
		solver = Solver(model, data0,
		            print_every=400, num_epochs=80, batch_size=400,
		            update_rule='sgd',
		            optim_config={
		              'learning_rate': learning_rate,
		            }
		     )
		solver.train()

		nnpredict = predictor(solver.model.loss)
		pickle.dump(nnpredict, open( "nnpredict."+filename+".p", "wb" ) )


		np = 4
		ms = data0['X_train'].shape[0]
		results = Parallel(n_jobs=-1,backend="multiprocessing", verbose=5)((delayed(uap)( data0['X_train'][ii*int(ms/np):(ii+1)*int(ms/np),:])) for ii in range(np))
		pickle.dump( results, open(  "uap."+filename+".p", "wb" ) )

	else:
		#data0 = pickle.load(open( "data."+filename+".p", "rb" ))
		data0 = pickle.load( open( "mnist.p", "rb" ) )
		nnpredict = pickle.load(open( "nnpredict."+filename+".p", "rb" ))
		results = pickle.load(open( "uap."+filename+".p", "rb" ))

	for count, r in enumerate(results):
		if count == 0:
			uapv = r
		else:
			uapv = numpy.vstack((uapv,r))

	#Compute SVD on uap
	uapv = normalize(uapv, axis=1, norm='l2')
	m,n = uapv.shape
	u,s,v = numpy.linalg.svd(uapv)

	#Compute SVD on random matrix
	randm = normalize(numpy.random.randn(m,n), axis=1, norm='l2')
	ur,sr,vr = numpy.linalg.svd(randm)

	# plt.semilogy(s)
	# plt.semilogy(sr)
	# plt.show()

	# #x = lena()
	# u, s, v = numpy.linalg.svd(uapv, full_matrices=False)
	# i = 1
	# pylab.figure()
	# pylab.ion()
	# print len(uapv)
	# while i < uapv.shape[1] - 1:
	# 	y = low_rank_approx((u, s, v), r=i)
	# 	pylab.imshow(y, cmap=pylab.cm.gray)
	# 	pylab.draw()
	# 	i += 1
	# 	#print percentage of singular spectrum used in approximation
	# 	print "%0.2f %%" % (100 * i / uapv.shape[1])
	# 	print numpy.linalg.norm(uapv-y)


	vfool = numpy.zeros(n)
	for ii in range(10):
		vfool = vfool + v[ii]

	print numpy.linalg.norm(vfool)
	print numpy.linalg.norm(data0['X_train'][0])

	foolrate = Err(data0['X_train'],vfool)
	print foolrate

	foolrate = Err(data0['X_train'],numpy.random.randn(n))
	print foolrate


