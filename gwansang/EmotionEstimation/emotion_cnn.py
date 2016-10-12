#Ryan Shin: sungjin7127@gmail.com
#Date: 160927
#Obj: find the best model for emotion recogntion

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
print ("Packages loaded")

#Load data
cwd = os.getcwd()
loadpath = cwd + "/EmotionEstimation/data_gray.npz"
l = np.load(loadpath)

#Check what's included
print (l.files)

# Parse data
trainimg = l['trainimg']
trainlabel = l['trainlabel']
testimg = l['testimg']
testlabel = l['testlabel']
imgsize = l['imgsize']
#use_gray = l['use_gray']
ntrain = trainimg.shape[0]
nclass = trainlabel.shape[1]
dim    = trainimg.shape[1]
ntest  = testimg.shape[0]
print ("%d train images loaded" % (ntrain))
print ("%d test images loaded" % (ntest))
print ("%d dimensional input" % (dim))
print ("Image size is %s" % (imgsize))
print ("%d classes" % (nclass))

tf.set_random_seed(0)
n_input = dim
n_output = nclass


weight_list = []

def conv_basic(_input ):
    _w = {
        'wc1' : tf.Variable(tf.random_normal([3,3,1,96], stddev=0.1)),
        'wc2' : tf.Variable(tf.random_normal([3,3,96,96], stddev=0.1)),
        'wd1' : tf.Variable(tf.random_normal(
            [(int)(imgsize[0]/4*imgsize[1]/4)*96, 96], stddev=0.1)),
        'wd2' : tf.Variable(tf.random_normal([96, n_output], stddev=0.1))
    }
    
    weight_list.append(_w['wc1'] )
    weight_list.append(_w['wc2'] )
    weight_list.append(_w['wd1'] )
    weight_list.append(_w['wd2'] )

    _b = {
        'bc1' : tf.Variable(tf.random_normal([96], stddev=0.1)),
        'bc2' : tf.Variable(tf.random_normal([96], stddev=0.1)),
        'bd1' : tf.Variable(tf.random_normal([96], stddev=0.1)),
        'bd2' : tf.Variable(tf.random_normal([n_output], stddev=0.1)),
    }
    
    weight_list.append(_b['bc1'])
    weight_list.append(_b['bc2'])
    weight_list.append(_b['bd1'])
    weight_list.append(_b['bd2'])
    
    # INPUT
    _input_r = tf.reshape(_input, shape=[-1, imgsize[0], imgsize[1], 1])
    # CONVOLUTION LAYER 1
    _conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_input_r
        , _w['wc1'], strides=[1, 1, 1, 1], padding='SAME'), _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1]
        , strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, 1.0)
    # CONVOLUTION LAYER 2
    _conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_pool_dr1
        , _w['wc2'], strides=[1, 1, 1, 1], padding='SAME'), _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1]
        , strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, 1.0 )
    # VECTORIZE
    _dense1 = tf.reshape(_pool_dr2
                         , [-1, _w['wd1'].get_shape().as_list()[0]])
    # FULLY CONNECTED LAYER 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, 1.0)
    # FULLY CONNECTED LAYER 2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    # RETURN
    out = {
        'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1
        , 'pool1_dr1': _pool_dr1, 'conv2': _conv2, 'pool2': _pool2
        , 'pool_dr2': _pool_dr2, 'dense1': _dense1, 'fc1': _fc1
        , 'fc_dr1': _fc_dr1, 'out': _out
    }
    return _out
print ("NETWORK READY")

#Define function

# tf Graph input
#x = tf.placeholder(tf.float32, [None, n_input])
#y = tf.placeholder(tf.float32, [None, n_output])
#keepratio = tf.placeholder(tf.float32)

# Functions!
#_pred = conv_basic(x, weights, biases, keepratio)['out']


