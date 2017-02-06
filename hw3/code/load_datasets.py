import numpy as np
import numpy.matlib

def makeSpiralSet(N=100,D=2,K=3):
	X = np.zeros((N*K,D)) # data matrix (each row = single example)
	y = np.zeros(N*K, dtype='uint8') # class labels
	for j in xrange(K):
	  ix = range(N*j,N*(j+1))
	  r = np.linspace(0.0,1,N) # radius
	  t = np.linspace(j*4,(j+1)*4,N) #+ np.random.randn(N)*0.2 # theta
	  X[ix] = np.c_[r*np.sin(t), (np.tile(r*np.cos(t),(D-1,1))).T]
	  y[ix] = j
	data = {
     'X_train': X, 'y_train': y,
     'X_val': X, 'y_val': y,
     'X_test': X, 'y_test': y}
	return data

def makeRandSet(N=100,D=2,K=3):
	X = np.zeros((N*K,D)) # data matrix (each row = single example)
	y = np.zeros(N*K, dtype='uint8') # class labels
	X_val = np.zeros((N*K,D)) # data matrix (each row = single example)
	y_val = np.zeros(N*K, dtype='uint8') # class labels
	X_test = np.zeros((N*K,D)) # data matrix (each row = single example)
	y_test = np.zeros(N*K, dtype='uint8') # class labels
	for j in xrange(K):
	  ix = range(N*j,N*(j+1))
	  r = np.linspace(0.0,1,N) # radius
	  t = np.linspace(j*4,(j+1)*4,N) #+ np.random.randn(N)*0.2 # theta
	  X[ix] = np.random.rand(N,D)
	  y[ix] = j
	  X_val[ix] = np.random.rand(N,D)
	  y_val[ix] = j
	  X_test[ix] = np.random.rand(N,D)
	  y_test[ix] = j
	data = {
     'X_train': X, 'y_train': y,
     'X_val': X_val, 'y_val': y_val,
     'X_test': X_test, 'y_test': y_test}
	return data


def load2Ddata(name):
	train = np.loadtxt('data/data'+name+'_train.csv')
	test = np.loadtxt('data/data'+name+'_test.csv')
	validate = np.loadtxt('data/data'+name+'_validate.csv')
	return train, test, validate

def make2Ddata(name):
	train, test, val = load2Ddata(name)
	X_train = train[:,0:2]
	y_train = train[:,2:3]
	y_train[y_train < 0] = 0
	y_train = np.squeeze(y_train.astype(int))
	X_test = test[:,0:2]
	y_test = test[:,2:3]
	y_test[y_test < 0] = 0
	y_test = np.squeeze(y_test.astype(int))
	X_val = val[:,0:2]
	y_val = val[:,2:3]
	y_val[y_val < 0] = 0
	y_val = np.squeeze(y_val.astype(int))

	data = {
     'X_train': X_train, 'y_train': y_train,
     'X_val': X_val, 'y_val': y_val,
     'X_test': X_test, 'y_test': y_test}
	return data

def loadMNIST(name):
	#print 'data/mnist_digit_'+str(name)+'.csv'
	x = np.loadtxt('data/mnist_digit_'+str(name)+'.csv')
	#print x.shape
	n,m = x.shape
	x = 2*x/255-1
	y = np.squeeze((np.ones(n)*int(str(name))).astype(int))
	return x, y


def makeMNIST(lname, ntrain, nval, ntest):

	for count, name in enumerate(lname):
		xall, yall = loadMNIST(name)
		if count == 0:
			X_train = xall[:ntrain, :]
			y_train = yall[:ntrain]
			X_val = xall[ntrain:ntrain+nval, :]
			y_val = yall[ntrain:ntrain+nval]
			X_test = xall[ntrain+nval:ntrain+nval+ntest, :]
			y_test = yall[ntrain+nval:ntrain+nval+ntest]
		else:
			X_train = np.vstack((X_train, xall[:ntrain, :]))
			y_train = np.hstack((y_train, yall[:ntrain]))
			X_val = np.vstack((X_val, xall[ntrain:ntrain+nval, :]))
			y_val = np.hstack((y_val, yall[ntrain:ntrain+nval]))
			X_test = np.vstack((X_test, xall[ntrain+nval:ntrain+nval+ntest, :]))
			y_test = np.hstack((y_test, yall[ntrain+nval:ntrain+nval+ntest]))

	data = {
     'X_train': X_train, 'y_train': y_train,
     'X_val': X_val, 'y_val': y_val,
     'X_test': X_test, 'y_test': y_test}
	return data