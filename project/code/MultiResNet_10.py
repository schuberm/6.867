#DenseNet
#Load necessary libraries
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
#%matplotlib inline

time1 = time.time()

# Load CIFAR10 dataset
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

currentCifar = 1
cifar = unpickle('./cifar10/data_batch_1')
cifarT = unpickle('./cifar10/test_batch')

total_layers = 10 #Specify how deep we want our network
units_between_stride = total_layers / 2


def denseBlock(input_layer,i,j):
    with tf.variable_scope("dense_unit"+str(i)):
        nodes = []
        a = slim.conv2d(input_layer,64,[3,3],normalizer_fn=slim.batch_norm)
        nodes.append(a)  # take record of block output
        for z in range(j):
            b = slim.conv2d(tf.concat(3,nodes),64,[3,3],normalizer_fn=slim.batch_norm)
            nodes.append(b)  #add connection between layers and match dimension
        return b

tf.reset_default_graph()

input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input')
label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
label_oh = slim.layers.one_hot_encoding(label_layer,10)

layer1 = slim.conv2d(input_layer,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
for i in range(5):
    layer1 = denseBlock(layer1,i,units_between_stride)
    layer1 = slim.conv2d(layer1,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
    
top = slim.conv2d(layer1,10,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')

output = slim.layers.softmax(slim.layers.flatten(top))

loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, reduction_indices=[1]))
trainer = tf.train.AdamOptimizer(learning_rate=0.001)
update = trainer.minimize(loss)


# Training and testing
init = tf.initialize_all_variables()
batch_size = 64
currentCifar = 1
total_steps = 10000
l = []
a = []
aT = []
with tf.Session() as sess:
    sess.run(init)
    i = 0
    draw = range(10000)
    while i < total_steps:
        if i % (10000/batch_size) != 0:
            batch_index = np.random.choice(draw,size=batch_size,replace=False)
        else:
            draw = range(10000)
            if currentCifar == 5:
                currentCifar = 1
                print "Switched CIFAR set to " + str(currentCifar)
            else:
                currentCifar = currentCifar + 1
                print "Switched CIFAR set to " + str(currentCifar)
            cifar = unpickle('./cifar10/data_batch_'+str(currentCifar))
            batch_index = np.random.choice(draw,size=batch_size,replace=False)
        x = cifar['data'][batch_index]
        x = np.reshape(x,[batch_size,32,32,3],order='F')
        x = (x/256.0)
        x = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
        y = np.reshape(np.array(cifar['labels'])[batch_index],[batch_size,1])
        _,lossA,yP,LO = sess.run([update,loss,output,label_oh],feed_dict={input_layer:x,label_layer:np.hstack(y)})
        accuracy = np.sum(np.equal(np.hstack(y),np.argmax(yP,1)))/float(len(y))
        l.append(lossA)
        a.append(accuracy)
        if i % 10 == 0: print "Step: " + str(i) + " Loss: " + str(lossA) + " Accuracy: " + str(accuracy)
        if i % 100 == 0: 
            point = np.random.randint(0,10000-500)
            xT = cifarT['data'][point:point+500]
            xT = np.reshape(xT,[500,32,32,3],order='F')
            xT = (xT/256.0)
            xT = (xT - np.mean(xT,axis=0)) / np.std(xT,axis=0)
            yT = np.reshape(np.array(cifarT['labels'])[point:point+500],[500])
            lossT,yP = sess.run([loss,output],feed_dict={input_layer:xT,label_layer:yT})
            accuracy = np.sum(np.equal(yT,np.argmax(yP,1)))/float(len(yT))
            aT.append(accuracy)
            print "Test set accuracy: " + str(accuracy)
        i+= 1

run_time = time.time()-time1
#Results
np.savetxt("MultiResNet10_time.txt", [run_time])
np.savetxt("MultiResNet10_training_loss.txt", l)
np.savetxt("MultiResNet10_training_acc.txt", a)
np.savetxt("MultiResNet10_test_acc.txt", aT)

plt.plot(l) # plot training loss
plt.xlabel("Steps")
plt.ylabel("Training Loss")
plt.title("MultiResNet Training Loss")
plt.savefig("MultiResNet10_training_loss.png")
plt.show()

plt.plot(a) # plot training accuracy
plt.xlabel("Steps")
plt.ylabel("Training Accuracy")
plt.title("MultiResNet Training Accuracy")
plt.savefig("MultiResNet10_training_accuracy.png")
plt.show()

plt.plot(aT) # plot test accuracy
plt.xlabel("Steps")
plt.ylabel("Testing Accuracy")
plt.title("MultiResNet Testing Accuracy")
plt.savefig("MultiResNet10_testing_accuracy.png")
plt.show()
np.max(aT) # report best test accuracy
