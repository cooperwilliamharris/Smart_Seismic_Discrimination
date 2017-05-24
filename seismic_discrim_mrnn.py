from __future__ import print_function
import tensorflow as tf
#from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn
import pandas 
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import timeit

#copyright Cooper W. Harris, University of Southern California, 2 May 2017

start = timeit.default_timer()

#minimum snr, in feature and data filenames from data prep code
target = 1.0

#get into data directory to read the data
os.chdir('../Data')
X = pandas.read_csv('X_feature_' + str(target) + 'snr.csv', header = None)
Y = pandas.read_csv('Y_label_' + str(target) + 'snr.csv', header = None)

#total number of seismograms being used
traces = len(X)

#ratio of total data to be used for testing (rest is for training)
divide = int(0.3*traces)

#all data
Xfull = X.as_matrix()

#all features
Yfull = Y.as_matrix()

#randomly split data into training and testing groups (new groups every run)
index_full = range(1,traces)
index_test = np.random.random_integers(1, traces-1, divide)
index_train = list(set(index_full) - set(index_test))

xtrain = Xfull[index_train]
ytrain = Yfull[index_train]
xtest = Xfull[index_test]
ytest = Yfull[index_test]

trained = len(xtrain)
tested = len(xtest)

###network parameters###

#num of datapoints per subwindow
chunk_size = 83
#number of subwindows (steps) for each time series
n_chunks = 12
#number of classification classes
n_classes = 2
#number of hidden layers
n_hidden = 70
#how many batches to train/test at once 
batch_size = traces-divide
#batch_size = int(batch_size*0.5)
#learning rate
learn_rate = 0.001
#number of runs through all chunks of all data
n_epochs = 500
#number of hidden layers in mrnn
num_layers = 4
#how often to print accuracy & loss to screen 
display_step = 10


###noisy expectation maximization parameters###
#add noise yes/no? 1=add nem noise; 0=no nem noise
add_noise = 0
#noise type (type of distribution drawn from to construct noise values)
#options: 0=uniform, 1=gaussian, 2=cauchy (aka: lorentz)
noise_type = 1
#noise variance
noise_var = 0.025
#noise annealing factor (decay)
annfactor = 0.2

###now define functions to call###

#tensorflow Graph input
x = tf.placeholder("float", [None, n_chunks, chunk_size])
y = tf.placeholder("float", [None, n_classes])

#define multilayer recurrent neural network
def RNN(x):

    layer = {'weights':tf.Variable(tf.random_normal([n_hidden,n_classes],stddev=0.15)),
             'biases':tf.Variable(tf.random_normal([n_classes]))}
    x = tf.unstack(x,n_chunks,1)
    
    #pick type of activation function for hidden layer
    #some choices: tf.tanh, tf.nn.relu, tf.sigmoid, tf.softplus
    activefn = tf.tanh
    lstm_cell = rnn.BasicLSTMCell(n_hidden,activation=activefn,forget_bias=1.0)    

    #include dropout
    lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=0.8, output_keep_prob=0.8) 

    #make the rnn a multilayer rnn
    lstm_cell = rnn.MultiRNNCell([lstm_cell]*num_layers)

    outputs, states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)   
    output = tf.matmul(outputs[-1],layer['weights'])+layer['biases']

    return output

prediction = RNN(x)

#define results plotter
#plots 2d plot of training error & training accuracy vs epoch
#also plots test performance as a histogram
#saves figure as '${figname}.pdf'
def plot(acc_list,loss_list,step_list,answer,guess,test_accuracy,step):
    figname = "latest_test"
    fig = plt.figure(1)
    plt.cla()
    ax = plt.subplot(3,1,1)
    plt.plot(step_list,loss_list,color='r')
    ax.set_ylabel('loss')    
    ax.set_xlabel('step')
    ax = plt.subplot(3,1,2)
    plt.plot(step_list,acc_list,color='m')
    ax.set_ylabel('accuracy')
    ax = plt.subplot(3,1,3)
    a = 2*guess+answer
    print(str(trained),"trained")
    print(str(tested),"tested")
    print(a)
    bins = ([0,1,2,3,4])
    b_all = np.histogram(a,bins)[0]
    b0 = int(b_all[0])
    print(str(b0),"missed explosions")
    b1 = int(b_all[1])
    print(str(b1),"missed earthquakes")
    b2 = int(b_all[2])
    print(str(b2),"found earthquakes")
    b3 = int(b_all[3])
    print(str(b3),"found explosions")
    names = ['false negative ' + str(b0),'false positive ' + str(b1),'true negative ' + str(b2),'true positive ' + str(b3)]
    plt.hist(a,bins,width=0.7,align='mid')
    plt.xticks(np.array(bins)+0.35,names,rotation=-40,fontsize=12)
    stop = timeit.default_timer()
    mark = stop-start
    mark = '%.3f' % mark
    perform = float('%.4f' % test_accuracy)
    perform = perform/0.01
    plt.title(str(perform) + '% Accurate, ' + str(mark) + ' seconds')
    ax.set_xlim(-0.25,4)
    ax.set_ylim(0,tested)
    text = str(step) + ' epochs'
    text2 = str(n_hidden) + ' hidden neurons'
    text3 = str(num_layers) + ' rnn layers'
    plt.text(0,80,text,fontsize=12,color='r')
    plt.text(0,65,text2,fontsize=12,color='r')
    plt.text(0,50,text3,fontsize=12,color='r')
    fig.set_tight_layout(True)
    os.chdir('../Figures/')   
    plt.savefig(figname + 'pdf', format ='pdf')   
    
#define training function that calls on the aforementioned RNN and plotting functions    
def train_nn(x):       
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    init = tf.global_variables_initializer()    

    #makes lists of training stats
    loss_list = []
    acc_list = []
    step_list = []
    
    #start tf.session
    with tf.Session() as sess:
        step = 0
        sess.run(init)
        
        #keep training until reach max iterations 
        while step < n_epochs:
            #reshape inputs 
            ind = np.random.random_integers(1, xtrain.shape[0]-1, batch_size)
            train_data = xtrain[ind]
            train_label = ytrain[ind]
            train_data = train_data.reshape((batch_size,n_chunks,chunk_size))
            sess.run(optimizer,feed_dict={x: train_data, y: train_label}) 
                                
            #Add NEM noise to output
            output_actv = tf.nn.softmax(prediction)
            ay = sess.run(output_actv, feed_dict={x: train_data})
            nv = noise_var/math.pow(step+1, annfactor)
            
            if noise_type == 0:
                noise = nv*(np.random.uniform(0,1,[batch_size, n_classes]))

            if noise_type == 1:    
                noise = nv*(np.random.normal(0,1,[batch_size, n_classes]))

            if noise_type == 2:    
                noise = nv*(np.random.standard_cauchy([batch_size, n_classes]))          
                
            #Filter noise to meet increased likelihood condition
            crit1 = noise*np.log(ay+ 1e-6)
            crit = crit1.sum(axis=1)
            index = (crit >= 0)
            noise_crit = np.repeat(index,n_classes)
            noise_index = np.reshape(noise_crit,[batch_size, n_classes])
            nem_noise = noise_index * noise * add_noise
            train_label = train_label + (nem_noise)

            #Train batches with noise in the training output
            #get accuracy
            acc, loss = sess.run([accuracy,cost],feed_dict={x: train_data, y: train_label})
            #get loss
#            loss = sess.run(cost,feed_dict={x: train_data, y: train_label})
            #append accuracy, loss, and step to lists tracking them
            acc_list.append(acc)
            loss_list.append(loss)
            step_list.append(step)
            #update step
            step += 1

            #display performance to screen (plot at end)           
            if step%display_step == 0:
                print("Step= " + str(step) + ", Iter= " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.4f}".format(acc))


        #after training is complete (based on epoch), test and plot performances        
        test_data = xtest[:divide]
        test_data = test_data.reshape((-1,n_chunks,chunk_size))
        test_label = ytest[:divide]
        test_accuracy,guess = sess.run([accuracy,correct],feed_dict={x: test_data, y: test_label})
        print("Testing Accuracy:" + "{:.4f}".format(test_accuracy))       
        answer = test_label[:,0]
        plot(acc_list,loss_list,step_list,answer,guess,test_accuracy,step)

                                                                        
train_nn(x)
plt.ioff()                                                                            
plt.show()                                                                            


    
