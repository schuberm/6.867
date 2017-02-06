import numpy as np
import gradient_check as gc
import loadParametersP1 as lp
import loadFittingDataP1 as lf

def gauss(x):
	gaussMean,gaussCov,quadBowlA,quadBowlb = lp.getData()
	fx = - 1/np.sqrt((2*np.pi)**x.shape[0]*np.linalg.det(gaussCov))*\
			np.exp(-0.5*np.dot((x-gaussMean).T,np.dot(np.linalg.inv(gaussCov),(x-gaussMean))))
	return fx

def grad_gauss(x):
	gaussMean,gaussCov,quadBowlA,quadBowlb = lp.getData()
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

def grad_descent(f, df, x0, lr=0.01, tol=1e-6):

	xold = x0
	while (np.linalg.norm(df(xold))>tol):
		xold = xold - lr*df(xold)
		print(xold)
		print(df(xold))

	return xold

def numeric_grad_descent(f, x0, lr=0.001, tol=1e-4):

	xold = x0
	while (np.linalg.norm(gc.eval_numerical_gradient(f, xold))>tol):
		xold = xold - lr*gc.eval_numerical_gradient(f, xold)
	return xold

def exact_batch_min():
	(X,y) = lf.getData()
	theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
	print(theta)
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

def batch_numeric_grad_descent(f, x):
	return numeric_grad_descent(f,x)

def sdg(f,xold, lr=0.01, tol=1e-4):

	t = 0.0
	(X,y) = lf.getData()
	i = np.random.randint(X.shape[0])
	lrold = lr
	while (True):
		#while (np.linalg.norm(gc.eval_numerical_gradient_i(f, xold, i))>tol):
		xold = xold - lr*gc.eval_numerical_gradient_i(f, xold, i)
		t += 1.0
		lr = lrold/t
		if i%25==0:
			i = 0
		else:
			i += 1
		if i%10==0:
			if (np.linalg.norm(gc.eval_numerical_gradient(J, xold))<tol):
				return xold

def sdg_minibatch(f,xold, lr=0.0001, tol=1e-4):

	t = 0.0
	(X,y) = lf.getData()
	i = xrange(1,50)
	lrold = lr
	while (True):
		#while (np.linalg.norm(gc.eval_numerical_gradient_i(f, xold, i))>tol):
		xold = xold - lr*gc.eval_numerical_gradient_i(f, xold, i)
		#t += 1.0
		#lr = lrold/t
		#if i%25==0:
		#	i = 0
		#else:
		#	i += 1
		#if i%10==0:
		if (np.linalg.norm(gc.eval_numerical_gradient_i(f, xold, i))<tol):
			return xold

x = (np.array([1.0,5.0]))
x0 = (np.array([1.1,5.0]))
#print(gauss(x))
#print(grad_gauss(x))
#print(-0.5*np.dot((x-gaussMean).T,np.dot(np.linalg.inv(gaussCov),(x-gaussMean))))
#print(gc.eval_numerical_gradient(gauss, x,h=0.00001))
#print(gc.eval_numerical_gradient_array(gauss, x, 1, h=1e-5))
#print(quadBowlA)
#print(np.dot(quadBowlA,x))
#print(grad_qbowl(x))
#print(gc.eval_numerical_gradient(qbowl,x,h=0.000001))
#grad_descent(qbowl,grad_qbowl,x0)

(X,y) = lf.getData()
#print(X[1,:])
theta = np.random.randn(X.shape[1])
theta = exact_batch_min() + 0.01*np.random.randn(X.shape[1])
J(theta)
#print(J(theta))
#theta_new = batch_numeric_grad_descent(J, theta)
#print(theta_new)
exact_batch_min()
#print(Ji(theta))
theta_new =sdg_minibatch(Ji_mini,theta)
print(theta_new)


#gc.eval_numerical_gradient_i(Ji, theta_new, 1)
#gc.eval_numerical_gradient(Ji_test, theta_new)