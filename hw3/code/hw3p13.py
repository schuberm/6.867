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
	weight_scale = 0.8
	learning_rate = 1e-2
	data3 = np.loadtxt('data/data_3class.csv')
	ntrain = 500
	ntest = 100
	nval = 100
	reg = 1.0
	figurename = './figures/p13.1l.10.1e-2.1.0.pdf'

	X_train = data3[0:ntrain,0:2]
	y_train = np.squeeze(data3[0:ntrain,2:3].astype(int))
	X_val = data3[ntrain:ntrain+nval,0:2]
	y_val = np.squeeze(data3[ntrain:ntrain+nval,2:3].astype(int))
	X_test = data3[ntrain+nval:ntrain+nval+ntest,0:2]
	y_test = np.squeeze(data3[ntrain+nval:ntrain+nval+ntest,2:3].astype(int))
	data0 = {
	      'X_train': X_train, 'y_train': y_train,
	      'X_val': X_val, 'y_val': y_val,
	      'X_test': X_test, 'y_test': y_test,
	    }

	dim = data0['X_train'].shape[1]

	model = FullyConnectedNet([10], input_dim=2, num_classes=3,
	          weight_scale=weight_scale, dtype=np.float64, reg=reg)
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
	plt.savefig(figurename)

	print solver.check_accuracy(data0['X_test'],data0['y_test'])