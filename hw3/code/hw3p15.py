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


if __name__=='__main__':

	#x = ld.loadMNIST('0')
	weight_scale = 0.01
	learning_rate = 1e-2
	rlMNIST = False
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


	model = FullyConnectedNet([300,100], input_dim=dim, num_classes=len(mnistlist),
	          weight_scale=weight_scale, dtype=np.float64)
	solver = Solver(model, data0,
	            print_every=80, num_epochs=40, batch_size=50,
	            update_rule='sgd',
	            optim_config={
	              'learning_rate': learning_rate,
	            }
	     )
	solver.train()

	plt.subplot(2, 1, 1)
	plt.title('Training loss')
	plt.plot(solver.loss_history, 'o')
	plt.xlabel('Iteration')

	plt.subplot(2, 1, 2)
	plt.title('Accuracy')
	plt.plot(solver.train_acc_history, '-o', label='train')
	plt.plot(solver.val_acc_history, '-o', label='val')
	plt.plot([0.5] * len(solver.val_acc_history), 'k--')
	plt.xlabel('Epoch')
	#plt.ylabel('Correct')
	plt.legend(loc='lower right',numpoints=1)
	#plt.gcf().set_size_inches(15, 12)
	plt.tight_layout()
	#plt.show()
	#plt.savefig(figurename)

	print (1-solver.check_accuracy(data0['X_test'],data0['y_test']))