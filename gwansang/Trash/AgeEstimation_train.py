
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


def search(dirname):
    search_result=[]
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename =os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.bmp':
            #print(full_filename[28:-4])
            search_result.append(full_filename)
        #print (full_filename)
    return search_result

def LoadImages(lable_file_name, image_path):
    Ages=[]
    ListAges=[]

    with open(lable_file_name, 'r') as f:
        reader = csv.reader(f, delimiter='\n') 
        for line in reader:
            [value]=line
            Ages.append(int(value))
            #print(value)
        
            if (int(value) in ListAges) == False:
                ListAges.append(int(value))
                #print(ListAges)
            '''
            bExist=False
            for ListAge in ListAges:
                if value == ListAge:
                    bExist=True
            if bExist==False:     
                ListAges.append(value)
            '''
        del value
    
    ListAges.sort()
    #print(np.size(Ages), np.ndim(Ages))
    #print(ListAges)
    ImageNameLists=search(image_path)
    ImageWidth=48
    ImageHeight=48
    
    ImageData=np.empty(shape=(np.size(ImageNameLists),ImageWidth,ImageHeight,1)) # 3 = [b, g, r]

    ReductionImageNameLists = ImageNameLists[:100000]
    for ImageName in ReductionImageNameLists:
        SourceImage=cv.imread(ImageName,0)
        ResizedImage=cv.resize(SourceImage, (ImageWidth,ImageHeight))
        ImageNum=int(ImageName[-9:-4])-1   
        
        ResizedImage=np.reshape(ResizedImage, [ImageWidth, ImageHeight, 1])
        #print(ImageNum)
        ImageData[ImageNum]=ResizedImage/255.0
        '''
        #print(ImageNum)
        if(ImageNum<20):
            fig=plt.figure()
            title='< Number: '+str(ImageNum)+', Value: '+str(Ages[ImageNum])+' >'
            plt.imshow(cv.cvtColor(ResizedImage, cv.COLOR_BGR2RGB))
            plt.axis('off')
        
            plt.title(title)
        '''    
    #print(ImageData.shape)

    OneHotAges=np.zeros((np.size(Ages),7), dtype=np.float32)
    index=0
    for Age in Ages:
        OneHotAges[index][ListAges.index(Age)]=1
        index=index+1

    #print(np.shape(OneHotAges), np.size(Ages), np.sum(OneHotAges))
    #print(OneHotAges)
    return ImageData, OneHotAges




# In[19]:

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

def model(X, w, w2, w3, w4, w5, w6, w7, w8, w9, w10, w14, w15, w_o, p_keep_conv, p_keep_hidden):
    
    # Conv2d
    # l1a shape=(?, 28, 28, 32)   48, 48, 64
    # padding='SAME' means output data's dimension is same as input image's. 
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1,1,1,1], padding='SAME'))
    l1b = tf.nn.relu(tf.nn.conv2d(l1a, w2, strides=[1,1,1,1], padding='SAME'))
    
    # Max pooling 
    # l1 shape=(?, 14, 14, 32)    48, 48, 64
    l1 = tf.nn.max_pool(l1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
        
    # Conv2d
    # l2a shape=(?, 14, 14, 64)    24, 24, 128
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w3, strides=[1,1,1,1], padding='SAME') )
    l2b = tf.nn.relu(tf.nn.conv2d(l2a, w4, strides=[1,1,1,1], padding='SAME') )  
    # l2 shape=(?, 7, 7, 64)      24, 24, 128
    l2 = tf.nn.max_pool(l2b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME' )
    l2 = tf.nn.dropout(l2, p_keep_conv)
    
    # l3a shape=(?, 7, 7, 128)    12, 12, 256
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w5, strides=[1, 1, 1, 1], padding='SAME'))
    l3b = tf.nn.relu(tf.nn.conv2d(l3a, w6, strides=[1, 1, 1, 1], padding='SAME'))
    l3c = tf.nn.relu(tf.nn.conv2d(l3b, w7, strides=[1, 1, 1, 1], padding='SAME'))    
    # l3 shape=(?, 4, 4, 128)     12, 12, 256 
    l3 = tf.nn.max_pool(l3c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_conv)
    
    
    # l4a shape=(?, 16, 16, 256)    6, 6 512
    l4a = tf.nn.relu(tf.nn.conv2d(l3, w8, strides=[1, 1, 1, 1], padding='SAME'))
    l4b = tf.nn.relu(tf.nn.conv2d(l4a, w9, strides=[1, 1, 1, 1], padding='SAME'))
    l4c = tf.nn.relu(tf.nn.conv2d(l4b, w10, strides=[1, 1, 1, 1], padding='SAME'))
    # l4 shape=(?, 8, 8, 256)     6, 6, 512
    l4 = tf.nn.max_pool(l4c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l4 = tf.nn.dropout(l4, p_keep_conv)
    
    l4 = tf.reshape(l4, [-1, w14.get_shape().as_list()[0]])   
    

    # Fully connected neural network
    l6 = tf.nn.relu(tf.matmul(l4, w14))
    l6 = tf.nn.dropout(l6, p_keep_hidden)
    l7 = tf.nn.relu(tf.matmul(l6, w15))
    l7 = tf.nn.dropout(l7, p_keep_hidden)
    
   
    pyx = tf.matmul(l7, w_o)
    
    return pyx


# In[10]:

lable_file_name='../Age_Data/Training_Dataset/Label.txt'
image_path='../Age_Data/Training_Dataset'


print "Data loading...."
inputX, inputY = LoadImages(lable_file_name, image_path)
print "Data Loaded"


Batch = tf.train.shuffle_batch( [inputX, inputY], batch_size=128, num_threads=4, capacity=50000, min_after_dequeue=10000, enqueue_many=True )
BatchX = tf.cast(Batch[0], tf.float32)
BatchY = tf.cast(Batch[1], tf.float32)
print "Data Shuffled"

# In[11]:
"""
TrainingRatio=0.9
TrainingSize=int(inputX.shape[0]*TrainingRatio)
TestSize=inputX.shape[0]-TrainingSize

trX=inputX[:TrainingSize]
trY=inputY[:TrainingSize]
teX=inputX[TrainingSize:TrainingSize+TestSize]
teY=inputY[TrainingSize:TrainingSize+TestSize]
"""

# In[12]:
"""
X=tf.placeholder(dtype=tf.float32, shape=[None,48,48,1])
Y=tf.placeholder(dtype=tf.float32, shape=[None,7])
"""

# In[20]:

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

w14 = init_weights([512 * 3 * 3, 4096], 'w14') # FC 128 * 4 * 4 inputs, 625 outputs
w15 = init_weights([4096, 1000], 'w15') # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([1000, 7], 'w_o')         # FC 625 inputs, 10 outputs (labels)

weight_list = [w, w2, w3, w4, w5, w6, w7, w8, w9, w10, w14, w15, w_o ]

p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)
py_x = model(BatchX, w, w2, w3, w4, w5, w6, w7, w8, w9, w10, w14, w15, w_o, p_keep_conv, p_keep_hidden)


# In[21]:

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, BatchY))
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)
#predict_op = tf.argmax(py_x, 1)

predict_op = tf.equal(tf.argmax(py_x, 1), tf.argmax(BatchY, 1))   
accuracy = tf.reduce_mean(tf.cast(predict_op, tf.float32))

saver = tf.train.Saver(weight_list)


print "model constructed"

# In[23]:

import time
batch_size = 32
test_size = 32




print "Session Opened"
sess = tf.Session()


sess.run(tf.initialize_all_variables())
coord = tf.train.Coordinator()
thread = tf.train.start_queue_runners( sess, coord )
print "start queue runner"

i= 0

try:
    saver.save(sess, "./Age_checkpoint_new.ckpt")
    print "check point saved"
    
    print "start training"
    while not coord.should_stop():
        [_, acc] = sess.run([train_op, accuracy], feed_dict={p_keep_conv:0.8, p_keep_hidden:0.5} )
        if acc > 0.95:
            saver.save(sess, "./Age_checkpoint_new.ckpt")
            break
        
        if i % 100 == 0:
            saver.save(sess, "./Age_checkpoint_new.ckpt")
            print "check point saved"
            print "step %d, accuracy %f" % (i, acc )
        i += 1
        
except tf.errors.OutOfRangeError:
    print ('Done training' )
    
finally:
    coord.request_stop()
    coord.join(thread)
    
        
sess.close()

print "Session closed"






