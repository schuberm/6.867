#from numpy import *
import numpy as np
from plotBoundary import *
import pylab as pl
import gradient_homebrew as gh
import scipy.optimize
from utils import *
from sklearn import linear_model, datasets
import warnings
#from utils import *
warnings.filterwarnings("ignore")

# import your LR training code

# parameters
name = '1'
print '======Training======'
# load data from csv files

C = [1e-2, 1e-1,1,10,100]
L = ['l1','l2']
Name = ['1','2','3','4']

EM = np.zeros([len(Name),len(C),len(L),2])
for ni,name in enumerate(Name):
	for nc, c in enumerate(C):
		for nl, l in enumerate(L):

			train = loadtxt('data/data'+name+'_train.csv')
			X = train[:,0:2]
			Y = train[:,2:3]

			logreg = linear_model.LogisticRegression(penalty=l,C=c)
			lr_obj = logreg.fit(X,Y)

			print(lr_obj.coef_)
			print(lr_obj.intercept_)
			print(lr_obj.n_iter_)
			# Define the predictLR(x) function, which uses trained parameters
			### TODO ###
			def constructPredictor(lr_obj):
				def predict(x):
					return sigmoid(lr_obj.predict(x))
				return predict

			predictLR = constructPredictor(lr_obj)

			n, m = X.shape
			corr = 0.0
			for i in range(n):
				#print(Y[i])
				#print(round(predictLR (X[i,:])))
				corr = corr + np.abs((Y[i]+1)/2-round(predictLR (X[i,:])))
			EM[ni,nc,nl,0] =corr/X.shape[0]
			print(corr/X.shape[0])
			# plot training results
			plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')
			pl.savefig('./figures/ps2_train_'+str(name)+'_'+str(l)+'_'+str(c)+'.pdf')
			pl.close()

			train = loadtxt('data/data'+name+'_test.csv')
			X = train[:,0:2]
			Y = train[:,2:3]
			n, m = X.shape
			corr = 0.0
			for i in range(n):
				#print(Y[i])
				#print(round(predictLR (X[i,:])))
				corr = corr + np.abs((Y[i]+1)/2-round(predictLR (X[i,:])))
			EM[ni,nc,nl,1] =corr/X.shape[0]
			print(corr/X.shape[0])
			# plot training results
			plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')
			pl.savefig('./figures/ps2_test_'+str(name)+'_'+str(l)+'_'+str(c)+'.pdf')
			pl.close()


print(EM)
np.save('EM_lr.txt',EM)

# print '======Validation======'
# # load data from csv files
# validate = loadtxt('data/data'+name+'_validate.csv')
# X = validate[:,0:2]
# Y = validate[:,2:3]

# # plot validation results
# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
# pl.show()
