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
from plotBoundary import *



if __name__=='__main__':

	weight_scale = 1.0
	learning_rate = 1e-2
	rl2d = True
	name = '2'
	figurename = './figures/hw2.2.1l.2.1.0.1e-2.pdf'
	figurenamebl = './figures/hw2.bound.2.1l.2.1.0.1e-2.pdf'
	

	if rl2d== True:
		data0 = ld.make2Ddata(name)
		pickle.dump( data0, open( "1.p", "wb" ) )
	else:
		data0 = pickle.load( open("1.p", "rb" ) )

	dim = data0['X_train'].shape[1]

	model = FullyConnectedNet([2], input_dim=dim, num_classes=2,
	          weight_scale=weight_scale, dtype=np.float64)
	solver = Solver(model, data0,
	            print_every=80, num_epochs=200, batch_size=50,
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
	plt.legend(loc='lower right',numpoints=1)
	plt.tight_layout()
	#plt.show()
	plt.savefig(figurename)

	scores = model.loss(data0['X_test'])


	def predictor (loss):
		def predict(x):
			return np.argmax(loss(x), axis=1)
		return predict

	nnpredict = predictor(solver.model.loss)

	print solver.check_accuracy(data0['X_test'],data0['y_test'])
   
	plotDecisionBoundary(data0['X_test'], data0['y_test'], nnpredict, [-1,0,1], title = 'NN')
	plt.title(['Test set accuracy = ' + str(solver.check_accuracy(data0['X_test'],data0['y_test']))])
	plt.savefig(figurenamebl)