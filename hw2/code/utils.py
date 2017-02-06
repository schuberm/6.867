import numpy as np
from plotBoundary import *
import pylab as pl
import gradient_homebrew as gh
import scipy.optimize
from scipy.spatial.distance import cdist,pdist,squareform
import time
import cvxopt
import cvxopt.solvers
from itertools import compress
# import your LR training code

## profiling decorator
def counted(fn):
    def wrapper(*args, **kwargs):
        wrapper.called+= 1
        before = time.time()
        retval = fn(*args, **kwargs)
        duration = time.time() - before
        wrapper.avgtime = ((wrapper.avgtime * wrapper.called) + duration) / (wrapper.called + 1)
        return retval
    wrapper.called = 0
    wrapper.avgtime = 0
    wrapper.__name__= fn.__name__
    return wrapper

def sigmoid(X):
    denom = 1.0 + np.exp(-1.0 * X)
    return 1.0 / denom

def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print f.__name__, 'took', end - start, 'time'
        return result
    return f_timer

@timefunc
def makeNLL(x,y,lmbda):
    def NLL(wn):
        n,m = x.shape
        w = wn[:m]
        b = wn[m]
        yp = (w.dot(x.T)+b)
        #y = np.squeeze(y)
        return sum(np.log(1+np.exp(np.squeeze(yp)*np.squeeze(y)*-1))) + lmbda*np.linalg.norm(w)**2
    return NLL    

def hingeLoss(X,Y,w,lmbda):
    n,m = X.shape
    tmp = 1-np.squeeze(Y)*np.squeeze(X.dot(w))
    #print(np.squeeze(Y).shape)
    #print(np.squeeze(X.dot(w)).shape)
    stmp = sum(tmp > 0.0)
    return lmbda*np.linalg.norm(w)/2 + 1/n*stmp

def khingeLoss(K,Y,w,lmbda):
    n,m = K.shape
    tmp = 1-np.squeeze(Y)*np.squeeze(K.dot(w))
    stmp = sum(tmp > 0.0)
    return lmbda*w.T.dot(K.dot(w))/2 + 1/n*stmp

def rbf(gamma,X):
    # n,m = X.shape
    # K = zeros((n,n));
    # for i in range(n):
    #     for j in range(n):
    #         K[i,j] = np.exp(-gamma*np.linalg.norm(X[i]-X[j])**2)
    K = np.exp(-gamma*squareform(pdist(X,'euclidean'))**2)
    return K

def rbfx(gamma,Xi,x):
    # n,m = Xi.shape
    # K = zeros(n);
    # for i in range(n):
    #     K[i] = np.exp(-gamma*np.linalg.norm(Xi[i]-x)**2)
    #print(Xi.shape)
    #print(np.matlib.repmat(x, Xi.shape[0],1))
    K = np.exp(-gamma*cdist(Xi,[x])**2)
    #print(K.shape)
    return K

def constructBPredictor(w,b):
    def predict(x):
        return w.dot(x)+b
    return predict

def constructLinearPredictor(w):
    def predict(x):
        return w[1:].dot(x)+w[0]
    return predict

def constructGaussianPredictor(w,gamma,Xi,Yi):
    #n,m = Xi.shape
    #K = zeros(n)
    def predict(x):
        #for i in range(n):
        #   K[i] = np.exp(-gamma*np.linalg.norm(Xi[i]-x)**2)
        K = rbfx(gamma,Xi,x)
        return sum(np.squeeze(w)*np.squeeze(K))
    return predict

def constructGaussianPredictorB(w,gamma,Xi,Yi,b):
    #n,m = Xi.shape
    #K = zeros(n);
    def predict(x):
        #for i in range(n):
        #   K[i] = np.exp(-gamma*np.linalg.norm(Xi[i]-x)**2)
        #K = np.exp(-gamma*cdist(Xi,np.matlib.repmat(x, Xi.shape[0],1))**2)
        K = rbfx(gamma,Xi,x)
        return sum(np.squeeze(w)*np.squeeze(Yi)*np.squeeze(K)) #+ b
    return predict

def train_lQPSVM(X,Y,K,C):
    n,m = K.shape
    P = cvxopt.matrix(np.outer(Y,Y) * K)
    #print(np.linalg.eigvals(P))
    q = cvxopt.matrix(np.ones(n) * -1)
    A = cvxopt.matrix(Y, (1,n))
    b = cvxopt.matrix(0.0)
    #G = cvxopt.matrix(np.diag(np.ones(n) * -1))
    #h = cvxopt.matrix(np.zeros(n))

    tmp1 = np.diag(np.ones(n) * -1)
    tmp2 = np.identity(n)
    G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(n)
    tmp2 = np.ones(n) * C
    h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

    tol = 1e-6
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.array(solution['x'])
    #print(alpha.shape)
    #print(X.shape)
    #alpha_1 = (alpha > tol) & (alpha < 0.98*C)
    alpha_1 = (alpha > tol) 
    #print(alpha[alpha_1])
    indl = list(compress(xrange(len(alpha_1)), alpha_1))
    w = (np.squeeze(Y)*np.squeeze(alpha)).dot(X)
    print(w.shape)
    # print(alpha.shape)
    # print(X.shape)
    # print(Y.shape)
    # print(X[alpha_1])
    b = 0
    for i in range(len(indl)):
        b = b + (1*Y[indl[i]]- w.dot(X[indl[i],:]))
    b = b/len(indl)
    nsv = len(indl)

    return w,b,nsv

def train_kQPSVM(X,Y,K,C,gamma):

    n,m = K.shape

    P = cvxopt.matrix(np.outer(Y,Y) * K)
    q = cvxopt.matrix(np.ones(n) * -1)
    A = cvxopt.matrix(Y, (1,n))
    b = cvxopt.matrix(0.0)

    tmp1 = np.diag(np.ones(n) * -1)
    tmp2 = np.identity(n)
    G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(n)
    tmp2 = np.ones(n) * C
    h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

    tol = 1e-6
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.array(solution['x'])
    #print(alpha.shape)
    #print(X.shape)
    alpha_1 = np.abs(alpha) > tol
    #print(alpha[alpha_1])
    indl = list(compress(xrange(len(alpha_1)), alpha_1))
    w = (Y*alpha).T.dot(X)

    #MASK
    Xi = X[indl,:]
    Yi = Y[indl]
    alphai = alpha[indl]
    #print(alphai)
    Ki = zeros(len(indl));
    for i in range(len(indl)):
        Ki[i] = np.exp(-gamma*np.linalg.norm(Xi[i]-X[indl[0]])**2)
    b = 1*Y[indl[0]]- sum(np.squeeze(alphai)*Ki)

    return alphai, Xi, Yi, b, alpha

### linear Pegasos ###
def lPegasos (X,Y,lmbda,max_epoch):
    n,m = X.shape
    t = 0
    eta = 0.1
    #lmbda = 0.002
    #wt = np.array([0.0, 0.0, 0.0])
    wt = np.zeros(X.shape[1]+1)
    epoch = 0
    #max_epoch = 100
    error = []
    while (epoch < max_epoch):
        for i in range(n):
            t = t + 1
            eta = 1/(t*lmbda)
            if (Y[i]*(wt[1:].dot(X[i,:])+wt[0]) < 1):
                wt[0] =  wt[0] + eta*Y[i]
                wt[1:] = (1-eta*lmbda)*wt[1:] + eta*Y[i]*X[i,:]
                #print(wt)
            else:
                #wt[0] =  wt[0] + eta*Y[i]
                #wt[1:]  = (1-eta*lmbda)*wt[1:]
                wt  = (1-eta*lmbda)*wt
                #print(wt)
        epoch = epoch + 1
        error.append(hingeLoss(X,Y,wt[1:],lmbda))
    tol = 1.00000001
    tmp = wt[1:].dot(X.T)+wt[0]
    nsv = tmp[np.abs(tmp)<tol].shape[0]
    return wt, error, nsv

### kernel Pegasos ###
def kPegasos(X,Y,K,lmbda,epochs):
    n,m = X.shape
    t = 0
    eta = 0.1
    wt = zeros(n)
    epoch = 0
    max_epoch = epochs
    error = []
    while (epoch < max_epoch):
        for i in range(n):
            t = t + 1
            eta = 1/(t*lmbda)
            cond = Y[i]*(wt.dot(K[:,i]))
            if (cond < 1):
                wt[i] = (1-eta*lmbda)*wt[i] + eta*Y[i]
                #print(wt)
            else:
                wt[i] = (1-eta*lmbda)*wt[i]
                #print(wt)
        epoch = epoch + 1
        error.append(khingeLoss(K,Y,wt,lmbda))
    return wt,error

def train_kPegasosSVM(X, Y, K, lmbda, epochs):

    alpha, error = kPegasos(X, Y, K, lmbda, epochs)

    #print(alpha)

    #MASK
    tol = 1e-6
    alpha_1 = np.abs(alpha) > tol
    #print(alpha[alpha_1])
    indl = list(compress(xrange(len(alpha_1)), alpha_1))
    Xi = X[indl,:]
    Yi = Y[indl]
    alphai = alpha[indl]
    return alphai, Xi, Yi, error

