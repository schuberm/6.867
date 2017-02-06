#HighwayNet
#Load necessary libraries
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#%matplotlib inline
import time

time1 = time.time()

#Load CIFAR10 dataset
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

currentCifar = 1
cifar = unpickle('./cifar10/data_batch_1')
cifarT = unpickle('./cifar10/test_batch')

total_layers = 25 #Specify how deep we want our network
units_between_stride = total_layers / 5

def highwayUnit(input_layer,i):
    with tf.variable_scope("highway_unit"+str(i)):
        H = slim.conv2d(input_layer,64,[3,3])
        T = slim.conv2d(input_layer,64,[3,3], #We initialize with a negative bias to push the network to use the skip connection
            biases_initializer=tf.constant_initializer(-1.0),activation_fn=tf.nn.sigmoid)
        output = H*T + input_layer*(1.0-T)
        return output

tf.reset_default_graph()

input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input')
label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
label_oh = slim.layers.one_hot_encoding(label_layer,10)

layer1 = slim.conv2d(input_layer,64,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
for i in range(5):
    for j in range(units_between_stride):
        layer1 = highwayUnit(layer1,j + (i*units_between_stride))
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
np.savetxt("HighwayNet_time.txt", [run_time])

np.savetxt("HighwayNet_training_loss.txt", l)
np.savetxt("HighwayNet_training_acc.txt", a)
np.savetxt("HighwayNet_test_acc.txt", aT)

plt.plot(l) # plot training loss
plt.xlabel("Steps")
plt.ylabel("Training Loss")
plt.title("HighwayNet Training Loss")
plt.show()
plt.savefig("HighwayNet_training_loss.png")

plt.plot(a) # plot training accuracy
plt.xlabel("Steps")
plt.ylabel("Training Accuracy")
plt.title("HighwayNet Training Accuracy")
plt.show()
plt.savefig("HighwayNet_training_accuracy.png")

plt.plot(aT) # plot test accuracy
plt.xlabel("Steps")
plt.ylabel("Testing Accuracy")
plt.title("HighwayNet Testing Accuracy")
plt.show()
plt.savefig("HighwayNet_testing_accuray.png")
np.max(aT) # report best test accuracy

