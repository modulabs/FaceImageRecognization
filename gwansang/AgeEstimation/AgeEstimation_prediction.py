
# coding: utf-8

# In[1]:

import cv2 as cv
import numpy as np
import os
import csv
import tensorflow as tf

import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')



weight_list = []

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

   
    
def model(X ):
    
    w = init_weights([3, 3, 1, 64], 'w')       # 3x3x1 conv, 32 outputs
    w2 = init_weights([3, 3, 64, 64], 'w2')     # 3x3x32 conv, 64 outputs

    w3 = init_weights([3, 3, 64, 128], 'w3')    # 3x3x32 conv, 128 outputs
    w4 = init_weights([3, 3, 128, 128], 'w4') # FC 128 * 4 * 4 inputs, 625 outputs

    w5 = init_weights([3, 3, 128, 256], 'w5') # FC 128 * 4 * 4 inputs, 625 outputs
    w6 = init_weights([3, 3, 256, 256], 'w6') # FC 128 * 4 * 4 inputs, 625 outputs
    w7 = init_weights([3, 3, 256, 256], 'w7') # FC 128 * 4 * 4 inputs, 625 outputs

    w8 = init_weights([3, 3, 256, 512], 'w8') # FC 128 * 4 * 4 inputs, 625 outputs
    w9 = init_weights([3, 3, 512, 512], 'w9') # FC 128 * 4 * 4 inputs, 625 outputs
    w10 = init_weights([3, 3, 512, 512], 'w10') # FC 128 * 4 * 4 inputs, 625 outputs

    # w11 = init_weights([3, 3, 512, 512], 'w11') # FC 128 * 4 * 4 inputs, 625 outputs
    # w12 = init_weights([3, 3, 512, 512], 'w12') # FC 128 * 4 * 4 inputs, 625 outputs
    # w13 = init_weights([3, 3, 512, 512], 'w13') # FC 128 * 4 * 4 inputs, 625 outputs

    w14 = init_weights([512 * 3 * 3, 4096], 'w14') # FC 128 * 4 * 4 inputs, 625 outputs
    w15 = init_weights([4096, 1000], 'w15') # FC 128 * 4 * 4 inputs, 625 outputs
    w_o = init_weights([1000, 7], 'w_o')         # FC 625 inputs, 10 outputs (labels)

    weight_list.append( w )
    weight_list.append( w2 )
    weight_list.append( w3 )
    weight_list.append( w4 )
    weight_list.append( w5 )
    weight_list.append( w6 )
    weight_list.append( w7 )
    weight_list.append( w8 )
    weight_list.append( w9 )
    weight_list.append( w10 )
    weight_list.append( w14 )
    weight_list.append( w15 )
    weight_list.append( w_o )
    

    
    # Conv2d
    # l1a shape=(?, 28, 28, 32)   48, 48, 64
    # padding='SAME' means output data's dimension is same as input image's. 
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1,1,1,1], padding='SAME'))
    l1b = tf.nn.relu(tf.nn.conv2d(l1a, w2, strides=[1,1,1,1], padding='SAME'))
    
    # Max pooling 
    # l1 shape=(?, 14, 14, 32)    48, 48, 64
    l1 = tf.nn.max_pool(l1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, 1.0)

    
    # Conv2d
    # l2a shape=(?, 14, 14, 64)    24, 24, 128
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w3, strides=[1,1,1,1], padding='SAME') )
    l2b = tf.nn.relu(tf.nn.conv2d(l2a, w4, strides=[1,1,1,1], padding='SAME') )  
    # l2 shape=(?, 7, 7, 64)      24, 24, 128
    l2 = tf.nn.max_pool(l2b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME' )
    l2 = tf.nn.dropout(l2, 1.0)

    
    # l3a shape=(?, 7, 7, 128)    12, 12, 256
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w5, strides=[1, 1, 1, 1], padding='SAME'))
    l3b = tf.nn.relu(tf.nn.conv2d(l3a, w6, strides=[1, 1, 1, 1], padding='SAME'))
    l3c = tf.nn.relu(tf.nn.conv2d(l3b, w7, strides=[1, 1, 1, 1], padding='SAME'))    
    # l3 shape=(?, 4, 4, 128)     12, 12, 256 
    l3 = tf.nn.max_pool(l3c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, 1.0)
    
    
    # l4a shape=(?, 16, 16, 256)    6, 6 512
    l4a = tf.nn.relu(tf.nn.conv2d(l3, w8, strides=[1, 1, 1, 1], padding='SAME'))
    l4b = tf.nn.relu(tf.nn.conv2d(l4a, w9, strides=[1, 1, 1, 1], padding='SAME'))
    l4c = tf.nn.relu(tf.nn.conv2d(l4b, w10, strides=[1, 1, 1, 1], padding='SAME'))
    # l4 shape=(?, 8, 8, 256)     6, 6, 512
    l4 = tf.nn.max_pool(l4c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l4 = tf.nn.dropout(l4, 1.0)
    
    l4 = tf.reshape(l4, [-1, w14.get_shape().as_list()[0]])   
    
    # Fully connected neural network
    l6 = tf.nn.relu(tf.matmul(l4, w14))
    l6 = tf.nn.dropout(l6, 1.0)
    l7 = tf.nn.relu(tf.matmul(l6, w15))
    l7 = tf.nn.dropout(l7, 1.0)
    
    return tf.matmul(l7, w_o)

#py_x = model(X, w, w2, w3, w4, w5, w6, w7, w8, w9, w10, w14, w15, w_o, p_keep_conv, p_keep_hidden)





