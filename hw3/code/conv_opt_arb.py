import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import sys
import math
import time

DATA_PATH = 'art_data/'
#DATA_FILE = DATA_PATH + 'art_data.pickle'
DATA_FILE = DATA_PATH + 'augmented_art_data.pickle'
IMAGE_SIZE = 50
NUM_CHANNELS = 3
NUM_LABELS = 11
INCLUDE_TEST_SET = False

class ArtistConvNet:
	def __init__(self, params, invariance=False):
		'''Initialize the class by loading the required datasets 
		and building the graph'''
		self.load_pickled_dataset(DATA_FILE)
		self.invariance = invariance
		if invariance:
			self.load_invariance_datasets()
		self.graph = tf.Graph()
		self.define_tensorflow_graph(params)

	def define_tensorflow_graph(self, params):
		print '\nDefining model...'
		
		batch_size = params['batch_size']
		learning_rate = params['learning_rate']
		num_training_steps = params['num_training_steps']
		train_size = self.train_Y.shape[0]

		# Enable dropout and weight decay normalization
		dropout_prob = 0.9 # set to < 1.0 to apply dropout, 1.0 to remove
		weight_penalty = 0.0005 # set to > 0.0 to apply weight penalty, 0.0 to remove

		with self.graph.as_default():
			# Input data
			tf_train_batch = tf.placeholder(
			    tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
			tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_LABELS))
			tf_valid_dataset = tf.constant(self.val_X)
			tf_test_dataset = tf.placeholder(
			    tf.float32, shape=[len(self.val_X), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
			tf_train_dataset = tf.placeholder(
				tf.float32, shape=[len(self.train_X), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

			# Implement dropout
			dropout_keep_prob = tf.placeholder(tf.float32)

			# Network weights/parameters that will be learned
			for count, l in enumerate(params['layers']):
				print count
				if count == 0:
					if l['hidden'] is False:
						l['weights'] = tf.Variable(tf.truncated_normal(
							[l['filter_size'], l['filter_size'], NUM_CHANNELS, l['depth']], stddev=0.1))
						l['biases'] = tf.Variable(tf.zeros(l['depth']))
						l['feat_map_size'] = int(math.ceil(float(IMAGE_SIZE) / l['stride']))
						if l['pooling']:
							l['feat_map_size'] = int(math.ceil(float(l['feat_map_size']) / l['pool_stride']))
				elif count == (len(params['layers'])-1):
					if l['hidden']:
						print l
						l['weights'] = tf.Variable(tf.truncated_normal(
						  [l['num_hidden'], NUM_LABELS], stddev=0.1))
						l['biases'] = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))
				#if count: #> 0:
				else:
					if l['hidden'] is True:
						print l
						if 'feat_map_size' in params['layers'][count-1]:
							print l
							l['weights'] = tf.Variable(tf.truncated_normal(
							[params['layers'][count-1]['feat_map_size'] * params['layers'][count-1]['feat_map_size'] * params['layers'][count-1]['depth'], l['num_hidden']], stddev=0.1))
							l['biases']  = tf.Variable(tf.constant(1.0, shape=[l['num_hidden']]))
						else:
							l['weights'] = tf.Variable(tf.truncated_normal(
							[params['layers'][count-1]['num_hidden'], l['num_hidden']], stddev=0.1))
							l['biases']  = tf.Variable(tf.constant(1.0, shape=[l['num_hidden']]))
					else:
						l['weights'] = tf.Variable(tf.truncated_normal(
							[l['filter_size'], l['filter_size'], params['layers'][count-1]['depth'], l['depth']], stddev=0.1))
						l['biases'] = tf.Variable(tf.zeros(l['depth']))
						l['feat_map_size'] = int(math.ceil(float(params['layers'][count-1]['feat_map_size'])/ l['stride']))
						if l['pooling']:
							l['feat_map_size'] = int(math.ceil(float(l['feat_map_size']) / l['pool_stride']))
			# Model
			def network_model(data):
				'''Define the actual network architecture'''

				for count, l in enumerate(params['layers']):
					if count == 0:
						conv = tf.nn.conv2d(data, l['weights'], [1, l['stride'], l['stride'], 1], padding='SAME')
						hidden = tf.nn.relu(conv + l['biases'])

						if l['pooling']:
							hidden = tf.nn.max_pool(hidden, ksize=[1, l['pool_filter_size'], l['pool_filter_size'], 1], 
									   strides=[1, l['pool_stride'], l['pool_stride'], 1],
                         			   padding='SAME', name='pool'+str(count+1))
							# hidden = tf.nn.lrn(hidden, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
       #              											name='norm2')
							hidden = tf.nn.local_response_normalization(hidden)

					elif (count > 0) and (count < len(params['layers'])-1):
						if l['hidden'] == False :
							conv = tf.nn.conv2d(hidden, l['weights'], [1, l['stride'], l['stride'], 1], padding='SAME')
							hidden = tf.nn.relu(conv + l['biases'])

							if l['pooling']:
								hidden = tf.nn.max_pool(hidden, ksize=[1, l['pool_filter_size'], l['pool_filter_size'], 1], 
									   strides=[1, l['pool_stride'], l['pool_stride'], 1],
                         			   padding='SAME', name='pool'+str(count+1))
								#hidden = tf.nn.lrn(hidden, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    			#								name='norm2')
								hidden = tf.nn.local_response_normalization(hidden)
						else:
							#print l
							shape = hidden.get_shape().as_list()
							ss = 1
							for s in shape[1:]:
								ss *=s
							#print ss
							#print shape[1] * shape[2] * shape[3]
							reshape = tf.reshape(hidden, [shape[0], ss])
							hidden = tf.nn.relu(tf.matmul(reshape, l['weights']) + l['biases'])
							hidden = tf.nn.dropout(hidden, dropout_keep_prob)
					else:
						print l
						output = tf.matmul(hidden, l['weights']) + l['biases']


				return output

			# Training computation
			logits = network_model(tf_train_batch)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

			# Add weight decay penalty
			for count, l in enumerate(params['layers']):
				if l['hidden']:
					loss = loss + weight_decay_penalty([l['weights']], weight_penalty)

			# Optimizer
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
			  # Decay once per epoch, using an exponential schedule starting at 0.01.
			batch = tf.Variable(0, dtype=tf.float32)
			learning_rate = tf.train.exponential_decay(
			      0.01,                # Base learning rate.
			      batch * batch_size,  # Current index into the dataset.
			      train_size,          # Decay step.
			      0.95,                # Decay rate.
			      staircase=True)
			optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss)

			# Predictions for the training, validation, and test data.
			batch_prediction = tf.nn.softmax(logits)
			train_prediction = tf.nn.softmax(network_model(tf_train_dataset))
			valid_prediction = tf.nn.softmax(network_model(tf_valid_dataset))
			test_prediction = tf.nn.softmax(network_model(tf_test_dataset))


			def train_model(num_steps=num_training_steps):
				'''Train the model with minibatches in a tensorflow session'''

				#saver = tf.train.Saver()
				with tf.Session(graph=self.graph) as session:
					tf.initialize_all_variables().run()

					saver = tf.train.Saver()
					print 'Initializing variables...'

					if params['reload']:
						  saver.restore(session, "model2.ckpt")
  						  print("Model restored.")

  					else:
					
						for step in range(num_steps):
							offset = (step * batch_size) % (self.train_Y.shape[0] - batch_size)
							batch_data = self.train_X[offset:(offset + batch_size), :, :, :]
							batch_labels = self.train_Y[offset:(offset + batch_size), :]
							
							# Data to feed into the placeholder variables in the tensorflow graph
							#feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, 
							#			 dropout_keep_prob: dropout_prob}
							feed_dict = {tf_train_batch : batch_data, tf_train_labels : batch_labels, 
										 dropout_keep_prob: dropout_prob}
							_, l, predictions = session.run(
							  [optimizer, loss, batch_prediction], feed_dict=feed_dict)
							if (step % 100 == 0):
								train_preds = session.run(train_prediction, feed_dict={tf_train_dataset: self.train_X,
												   dropout_keep_prob : 1.0})
								val_preds = session.run(valid_prediction, feed_dict={dropout_keep_prob : 1.0})
								print ''
								print('Batch loss at step %d: %f' % (step, l))
								print('Batch training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
								print('Validation accuracy: %.1f%%' % accuracy(val_preds, self.val_Y))
								print('Full train accuracy: %.1f%%' % accuracy(train_preds, self.train_Y))
						
						# This code is for the final question
					if self.invariance:
						print "\n Obtaining final results on invariance sets!"
						sets = [self.val_X, self.translated_val_X, self.bright_val_X, self.dark_val_X, 
								self.high_contrast_val_X, self.low_contrast_val_X, self.flipped_val_X, 
								self.inverted_val_X,]
						set_names = ['normal validation', 'translated', 'brightened', 'darkened', 
									 'high contrast', 'low contrast', 'flipped', 'inverted']
						
						for i in range(len(sets)):
							preds = session.run(test_prediction, 
								feed_dict={tf_test_dataset: sets[i], dropout_keep_prob : 1.0})
							print 'Accuracy on', set_names[i], 'data: %.1f%%' % accuracy(preds, self.val_Y)

							# save final preds to make confusion matrix
							if i == 0:
								self.final_val_preds = preds

					save_path = saver.save(session, "model2.ckpt")
	  				print("Model saved in file: %s" % save_path)
			# save train model function so it can be called later
			self.train_model = train_model

	def load_pickled_dataset(self, pickle_file):
		print "Loading datasets..."
		with open(pickle_file, 'rb') as f:
			save = pickle.load(f)
			self.train_X = save['train_data']
			self.train_Y = save['train_labels']
			self.val_X = save['val_data']
			self.val_Y = save['val_labels']

			if INCLUDE_TEST_SET:
				self.test_X = save['test_data']
				self.test_Y = save['test_labels']
			del save  # hint to help gc free up memory
		print 'Training set', self.train_X.shape, self.train_Y.shape
		print 'Validation set', self.val_X.shape, self.val_Y.shape
		if INCLUDE_TEST_SET: print 'Test set', self.test_X.shape, self.test_Y.shape

	def load_invariance_datasets(self):
		with open(DATA_PATH + 'invariance_art_data.pickle', 'rb') as f:
			save = pickle.load(f)
			self.translated_val_X = save['translated_val_data']
			self.flipped_val_X = save['flipped_val_data']
			self.inverted_val_X = save['inverted_val_data']
			self.bright_val_X = save['bright_val_data']
			self.dark_val_X = save['dark_val_data']
			self.high_contrast_val_X = save['high_contrast_val_data']
			self.low_contrast_val_X = save['low_contrast_val_data']
			del save  

def weight_decay_penalty(weights, penalty):
	return penalty * sum([tf.nn.l2_loss(w) for w in weights])

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

if __name__ == '__main__':
	invariance = False
	if len(sys.argv) > 1 and sys.argv[1] == 'invariance':
		print "Testing finished model on invariance datasets!"
		invariance = True
	
	t1 = time.time()
	params = {}
	params['batch_size'] = 240
	params['learning_rate'] = 0.01
	params['num_training_steps'] = 1001
	params['reload'] = True

	layer1 = {}
	layer2 = {}
	layer3 = {}
	layer4 = {}
	layer5 = {}
	layer6 = {}
	layer7 = {}

	layer1['filter_size'] = 5
	layer1['depth'] = 48
	layer1['stride'] = 2
	layer1['pooling'] = True
	layer1['hidden'] = False
	layer1['pool_filter_size'] = 2
	layer1['pool_stride'] = 2
	layer1['lnr'] = True

	layer2['filter_size'] = 5
	layer2['depth'] = 64
	layer2['stride'] = 2
	layer2['pooling'] = False
	layer2['hidden'] = False
	layer2['pool_filter_size'] = 2
	layer2['pool_stride'] = 2
	layer2['lnr'] = True

	layer3['filter_size'] = 5
	layer3['depth'] = 128
	layer3['stride'] = 2
	layer3['pooling'] = True
	layer3['hidden'] = False
	layer3['pool_filter_size'] = 2
	layer3['pool_stride'] = 2
	layer3['lnr'] = True

	layer7['filter_size'] = 3
	layer7['depth'] = 256
	layer7['stride'] = 2
	layer7['pooling'] = True
	layer7['hidden'] = False
	layer7['pool_filter_size'] = 2
	layer7['pool_stride'] = 2
	layer7['lnr'] = True

	layer4['hidden'] = True
	layer4['num_hidden'] = 1000
	############################################################
	#Last layer must have same number of nodes as previous layer
	############################################################
	layer5['hidden'] = True
	layer5['num_hidden'] = 128

	layer6['hidden'] = True
	layer6['num_hidden'] = 128

	params['layers'] = [layer1, layer2, layer3, layer4, layer5, layer6]

	conv_net = ArtistConvNet(params,invariance=invariance)
	conv_net.train_model()
	t2 = time.time()
	print "Finished training. Total time taken:", t2-t1
