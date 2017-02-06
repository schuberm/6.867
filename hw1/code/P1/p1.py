import numpy as np
import gradient_check as gc
import loadParametersP1 as lp
import loadFittingDataP1 as lf
import matplotlib.pyplot as plt

def gauss(x):
	gaussMean,gaussCov,quadBowlA,quadBowlb = lp.getData()
	#gaussCov = 0.01*gaussCov
	fx = - 1/np.sqrt((2*np.pi)**x.shape[0]*np.linalg.det(gaussCov))*\
			np.exp(-0.5*np.dot((x-gaussMean).T,np.dot(np.linalg.inv(gaussCov),(x-gaussMean))))
	#fx = np.exp(-0.5*np.dot((x-gaussMean).T,np.dot(np.linalg.inv(gaussCov),(x-gaussMean))))
	return fx

def grad_gauss(x):
	gaussMean,gaussCov,quadBowlA,quadBowlb = lp.getData()
	#gaussCov = 0.01*gaussCov
	dfx = -gauss(x)*np.dot(np.linalg.inv(gaussCov),(x-gaussMean))
	return dfx

def qbowl(x):
	gaussMean,gaussCov,quadBowlA,quadBowlb = lp.getData()
	fx = 0.5*np.dot(x.T,np.dot(quadBowlA,x))-np.dot(x.T,quadBowlb)
	return fx

def grad_qbowl(x):
	gaussMean,gaussCov,quadBowlA,quadBowlb = lp.getData()
	dfx = np.dot(quadBowlA,x)-quadBowlb
	return dfx

def qbowl_exact():
	gaussMean,gaussCov,quadBowlA,quadBowlb = lp.getData()
	x = np.dot(np.linalg.inv(quadBowlA),quadBowlb)
	return x

def grad_descent(f, df, x0, lr=0.01, tol=1e-6):

	normlist = []
	xold = x0
	while (np.linalg.norm(df(xold))>tol):
		xold = xold - lr*df(xold)
		print(xold)
		print(df(xold))
		normlist.append(np.linalg.norm(df(xold)))

	return xold, normlist

def analytic_grad_descent(df, x0, lr=0.001, tol=1e-6):

	normlist = []
	xold = x0
	#print(df(xold))
	theta_exact = np.array([10.0, 10.0])
	#theta_exact = qbowl_exact()
	while (np.linalg.norm(theta_exact-xold)>tol):
		xold = xold - lr*df(xold)
		#print(xold)
		print(df(xold))
		#normlist.append(np.linalg.norm(df(xold)))
		normlist.append(np.linalg.norm(theta_exact-xold))
		print(np.linalg.norm(theta_exact-xold))
		if len(normlist) > 200:
			return xold, normlist

	return xold, normlist

def numeric_grad_descent(f, x0, lr=0.001, tol=1e-4):

	xold = x0
	#print(gc.eval_numerical_gradient(f, xold))
	theta_exact = exact_batch_min()
	#theta_exact = np.array([10.0, 10.0])
	#theta_exact = qbowl_exact()
	normlist = []
	while (np.linalg.norm(theta_exact-xold)>tol):
		xold = xold - lr*gc.eval_numerical_gradient(f, xold, verbose=False)
		#print(xold)
		#normlist.append(np.linalg.norm(gc.eval_numerical_gradient(f, xold, verbose=False)))
		normlist.append(np.linalg.norm(theta_exact-xold))
		if len(normlist) > 200:
			return xold, normlist
	return xold, normlist

def exact_batch_min():
	(X,y) = lf.getData()
	theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
	return theta

def J(theta):
	(X,y) = lf.getData()
	return np.linalg.norm(X.dot(theta)-y)

def Ji(theta,i):
	(X,y) = lf.getData()
	#i=np.random.randint(X.shape[0])
	return (X[i,:].T.dot(theta)-y[i])**2

def Ji_mini(theta,i):
	(X,y) = lf.getData()
	#i=np.random.randint(X.shape[0])
	return np.linalg.norm(X[i,:].dot(theta)-y[i])

def Ji_test(theta):
	(X,y) = lf.getData()
	#i=np.random.randint(X.shape[0])
	return (X[1,:].T.dot(theta)-y[1])**2

def dJi_test(theta):
	(X,y) = lf.getData()
	return 2*X[1,:]*(X[1,:].T.dot(theta)-y[1])

def batch_numeric_grad_descent(f, x):
	return numeric_grad_descent(f,x)


#THIS WORKS!
# def sdg(f,xold, lr=0.0001, tol=1e-12):

# 	t = 0.0
# 	(X,y) = lf.getData()
# 	#i = np.random.randint(X.shape[0]-1)
# 	i = 1
# 	lrold = lr
# 	while (True):
# 		#while (np.linalg.norm(gc.eval_numerical_gradient_i(f, xold, i))>tol):
# 		xold = xold - lr*gc.eval_numerical_gradient_i(f, xold, i,verbose=False)
# 		t += 1.0
# 		lr = lrold/t
# 		if i%(X.shape[0]-1)==0:
# 			i = 0
# 		else:
# 			i += 1
# 		if t%100==0:
# 			print(xold)
# 			#print(exact_batch_min())
# 			#if (np.linalg.norm(gc.eval_numerical_gradient(J, xold))<tol):
# 			conv = np.sum([gc.eval_numerical_gradient_i(f, xold, i,verbose=False) for i in range(X.shape[0]-1)])
# 			if (conv<tol):
# 				return xold

#THIS WORKS!
def sdg(f, xold, lr=0.002, tol=1e-12):

	t = 0.0
	(X,y) = lf.getData()
	#i = np.random.randint(X.shape[0]-1)
	theta_exact = exact_batch_min() 
	normlist = []
	i = 1
	lrold = lr
	#conv = np.sum([gc.eval_numerical_gradient_i(f, xold, ii, verbose=False) for ii in range(X.shape[0]-1)])
	while (True):
		xnew = xold - lr*gc.eval_numerical_gradient_i(f, xold, i, verbose=False)
		t += 1.0
		lr = lrold/t
		#print(np.linalg.norm(xold-xnew))
		#normlist.append(abs(J(theta_exact)-J(xnew)))
		normlist.append(np.linalg.norm(theta_exact-xnew))
		if i%(X.shape[0]-1)==0:
			i = 0
		else:
			i += 1
		#if t%20==0:
			#print(xold)
			#conv = np.sum([gc.eval_numerical_gradient_i(f, xold, ii, verbose=False) for ii in range(X.shape[0]-1)])
			#conv = J(xold)
			#print(conv)
			#theta = exact_batch_min() 
		if (np.linalg.norm(xold-xnew) < 10.0e-7):
			print(xold)
			#conv = J(xold)
			#conv = np.sum([gc.eval_numerical_gradient_i(f, xold, ii, verbose=False) for ii in range(X.shape[0]-1)])
			#print(conv)
			#normlist.append(conv)
			normlist.append(np.linalg.norm(theta_exact-xnew))
			return xold, normlist
		if t == 200:
			return xold, normlist
		xold = xnew
		
	#conv = np.sum([gc.eval_numerical_gradient_i(f, xold, ii, verbose=False) for ii in range(X.shape[0]-1)])
	#normlist.append(conv)
	print(normlist)
	print(xold)
	return xold, normlist

def sdg_minibatch(f, xold, lr=0.0001, tol=1e-4):

	t = 0.0
	(X,y) = lf.getData()
	#i = xrange(1,50)
	lrold = lr
	normlist = []
	theta_exact = exact_batch_min() 
	while (True):
		#while (np.linalg.norm(gc.eval_numerical_gradient_i(f, xold, i))>tol):
		i = np.random.randint(100, size=10)
		xold = xold - lr*gc.eval_numerical_gradient_i(f, xold, i, verbose=False)
		t += 1.0
		#lr = lrold/t
		normlist.append(np.linalg.norm(theta_exact-xold))
		#if i%25==0:
		#	i = 0
		#else:
		#	i += 1
		#if i%10==0:
		if t > 200:
			return xold, normlist
	return xold, normlist

####Q1.1####

##gauss##
# lr=1e6
# thetanorm = np.linalg.norm(np.array([10.0,10.0]))
# x0 = np.array([13.0,9.000])
# xa,errora = analytic_grad_descent(grad_gauss,x0,lr=lr)
# nsteps = range(len(errora))
# plt.plot(nsteps,errora/thetanorm,'x',label='Analytical $\eta=$'+str(lr))
# #print(xa)
# xn,errorn = numeric_grad_descent(gauss,x0,lr=lr)
# nsteps = range(len(errorn))

# plt.plot(nsteps,errorn/thetanorm,label='Numerical $\eta=$'+str(lr))

# lr=1e5
# thetanorm = np.linalg.norm(np.array([10.0,10.0]))
# x0 = np.array([13.0,9.000])
# xa,errora = analytic_grad_descent(grad_gauss,x0,lr=lr)
# nsteps = range(len(errora))
# plt.plot(nsteps,errora/thetanorm,'x',label='Analytical $\eta=$'+str(lr))
# #print(xa)
# xn,errorn = numeric_grad_descent(gauss,x0,lr=lr)
# nsteps = range(len(errorn))

# plt.plot(nsteps,errorn/thetanorm,label='Numerical $\eta=$'+str(lr))

# lr=1e4
# thetanorm = np.linalg.norm(np.array([10.0,10.0]))
# x0 = np.array([13.0,9.000])
# xa,errora = analytic_grad_descent(grad_gauss,x0,lr=lr)
# nsteps = range(len(errora))
# plt.plot(nsteps,errora/thetanorm,'x',label='Analytical $\eta=$'+str(lr))
# #print(xa)
# xn,errorn = numeric_grad_descent(gauss,x0,lr=lr)
# nsteps = range(len(errorn))

# plt.plot(nsteps,errorn/thetanorm,label='Numerical $\eta=$'+str(lr))

##qbowl###
# lr=0.0001
# thetanorm = np.linalg.norm(qbowl_exact())
# x0 = np.array([100.5,100.001])
# xa,errora = analytic_grad_descent(grad_qbowl,x0,lr=lr)
# nsteps = range(len(errora))
# plt.plot(nsteps,errora/thetanorm,'x',label='Analytical $\eta=$'+str(lr))
# #print(xa)
# xn,errorn = numeric_grad_descent(qbowl,x0,lr=lr)
# nsteps = range(len(errorn))

# plt.plot(nsteps,errorn/thetanorm,label='Numerical $\eta=$'+str(lr))

# lr=0.001
# thetanorm = np.linalg.norm(qbowl_exact())
# x0 = np.array([100.5,100.001])
# xa,errora = analytic_grad_descent(grad_qbowl,x0,lr=lr)
# nsteps = range(len(errora))
# plt.plot(nsteps,errora/thetanorm,'x',label='Analytical $\eta=$'+str(lr))
# #print(xa)
# xn,errorn = numeric_grad_descent(qbowl,x0,lr=lr)
# nsteps = range(len(errorn))

# plt.plot(nsteps,errorn/thetanorm,label='Numerical $\eta=$'+str(lr))

# lr=0.01
# thetanorm = np.linalg.norm(qbowl_exact())
# x0 = np.array([100.5,100.001])
# xa,errora = analytic_grad_descent(grad_qbowl,x0,lr=lr)
# nsteps = range(len(errora))
# plt.plot(nsteps,errora/thetanorm,'x',label='Analytical $\eta=$'+str(lr))
# #print(xa)
# xn,errorn = numeric_grad_descent(qbowl,x0,lr=lr)
# nsteps = range(len(errorn))

# plt.plot(nsteps,errorn/thetanorm,label='Numerical $\eta=$'+str(lr))


#print(grad_gauss(x0))
#print(-0.5*np.dot((x-gaussMean).T,np.dot(np.linalg.inv(gaussCov),(x-gaussMean))))
#print(gc.eval_numerical_gradient(gauss, x0,h=0.00001))
#print(gc.eval_numerical_gradient_array(gauss, x, 1, h=1e-5))
#print(quadBowlA)
#print(np.dot(quadBowlA,x))
#print(grad_qbowl(x))
#print(gc.eval_numerical_gradient(qbowl,x,h=0.000001))
#grad_descent(qbowl,grad_qbowl,x0)

# N = np.array(range(0,20))
# L = np.array(range(0,20))
# sse = np.zeros((N.shape[0],L.shape[0]))
# for n in range(N.shape[0]):
# 	for l in range(L.shape[0]):
# 		sse[n,l] = gauss(np.array([N[n],L[l]]))

# plt.imshow(sse, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()

# ####Q1.2BGD####
# (X,y) = lf.getData()
# lr=0.001
# #print(X[1,:])
# thetanorm = np.linalg.norm(exact_batch_min())
# #lr should be ~0.001
# #theta = exact_batch_min() + 5.0*np.random.randn(X.shape[1])
# theta = 5.0*np.random.randn(X.shape[1])
# theta_new, error = batch_numeric_grad_descent(J, theta)
# nsteps = np.array(range(len(error)))#*100

# batch_plot, = plt.plot(nsteps,error/thetanorm,'o',label='BDG $\eta=$'+str(lr))
# #plt.yscale('log')
# #plt.savefig('../../figures/ps1_batch.pdf')
# #plt.show()

# theta_new, error_sdg = sdg(Ji,theta,lr=lr)
# nsteps = range(len(error_sdg))
# sdg_plot, = plt.plot(nsteps,error_sdg/thetanorm,'s',label='SDG $\eta=$'+str(lr))


# ####Q1.4minSGD####
# # (X,y) = lf.getData()
# # theta = np.random.randn(X.shape[1])
# # theta = exact_batch_min() + 1.0*np.random.randn(X.shape[1])
# # for lr in [0.00001,0.0001,0.001]:
# # 	theta_new, error_sdg = sdg_minibatch(Ji_mini,theta,lr=lr)
# # 	nsteps = range(len(error_sdg))
# # 	sdg_plot, = plt.plot(nsteps,error_sdg/thetanorm,'s',label=str(lr))

# theta_new, error_sdg = sdg_minibatch(Ji_mini,theta,lr=lr)
# nsteps = range(len(error_sdg))
# sdg_plot, = plt.plot(nsteps,error_sdg/thetanorm,'s',label='SDG_mini $\eta=$'+str(lr))


# plt.legend(numpoints=1,loc='lower left', fontsize = 'x-small')
# plt.yscale('log')
# plt.xlabel('Iterations')
# plt.ylabel('Error')
# plt.savefig('../../figures/ps1_minibatchcompare_initial.pdf')

print(qbowl_exact())