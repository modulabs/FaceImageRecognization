from __future__ import division
import skimage
import skimage.io
import skimage.transform

import scipy as scp
import scipy.misc
import pylab as plt
import numpy as np
import tensorflow as tf
import time
import os

import VGG_RGB
import cv2
import dlib

batchSize = 1
imageSize = 224
AgeImageSize = 24
EmotionImageSize = 48

epochNum = 0
num_class = 2
learning_rate = 1e-6


font = cv2.FONT_HERSHEY_SIMPLEX

directoryName = "face data/large"
#checkpoint_dir = "cps_vgg_rgb/"

# check points
GenderCheckPoint_dir = "GenderCKPT/"
AgeCheckPoint_dir = "AgeCKPT/"
EmotionCheckPoint_dir = "EmotionCKPT/"

TestImages = np.zeros([1, imageSize, imageSize, 3])

# female = 1, male = 0

# Open Sessions
GenderRecognitionSession = tf.Session() 
AgeEstimationSession = tf.Session()
EmotionEstimationSession = tf.Session()

vgg_rgb = VGG_RGB.VGG_RGB()

# create image space 
GenderValidateImages = np.zeros([1, imageSize, imageSize, 3], dtype=np.float32)
AgeValidateImages = np.zeros([1, AgeImageSize, AgeImageSize ], dtype=np.float32)
EmotionValidateImages = np.zeros([1, EmotionImageSize, EmotionImageSize], dtype=np.float32)

# GenderRecognision input arguments
X = tf.placeholder("float", [batchSize, imageSize, imageSize, 3])
Y = tf.placeholder("int32", [batchSize, 1])
train = tf.placeholder(tf.bool)

# AgeEstimation input arguments
AgeInputImage = tf.placeholder(tf.float32, [batchSize, AgeImageSize, AgeImageSize] )


#EmotionEstimation input arguments
EmotionInputImage = tf.placeholder( tf.float32, [batchSize, EmotionImageSize, EmotionImageSize ] )


# building each model 
vgg_rgb.build(X, Y, train, scope="VGG_RGB",  num_classes=num_class, debug=True)
AgePred = AgeEstimation( AgeInputImage )
EmotionPred = EmotionEstimation( EmotionInputImage )


GenderRecognitionSaver = tf.train.Saver()
AgeEstimationSaver = tf.train.Saver()
EmotionEstimationSaver = tf.train.Saver()

#ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
Genderckpt = tf.train.get_checkpoint_state(GenderCheckPoint_dir)
Ageckpt = tf.train.get_checkpoint_state(AgeCheckPoint_dir)
Emotionckpt = tf.train.get_checkpoint_state(EmotionCheckPoint_dir)



if Genderckpt and Genderckpt.model_checkpoint_path:
    GenderRecognitionSaver.restore(GenderRecognitionSession, Genderckpt.model_checkpoint_path)
    print ('load Gender model') 
    
if Ageckpt and Ageckpt.model_checkpoint_path:
    AgeEstimationSaver.restore(AgeEstimationSession, Ageckpt.model_checkpoint_path)
    print ('load Age model' )

if Emotionckpt and Emotionckpt.model_checkpoint_path:
    EmotionEstimationSaver.restore(EmotionEstimationSession, Emotionckpt.model_checkpoint_path)
    print ('load emotion model' )

    
    
print "camera start =========================="
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()



while(True):

    # Capture frame-by-frame
    time1 = time.time()
    ret, frame = cap.read()

    time2 = time.time()
    dets = detector(frame, 1)
    time3 = time.time()
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        i, d.left(), d.top(), d.right(), d.bottom()))
        x = d.left()
        y = d.top()
        x2 = d.right()
        y2 = d.bottom()
        crop_img = frame[y:y2,x:x2]

        
        # resizing images
        Gender_resized_img = cv2.resize(crop_img, (imageSize, imageSize)) 
        Age_resized_img = cv2.resize(crop_img, (AgeImageSize, AgeImageSize ))
        Emotion_resized_img = cv2.resize(crop_img, (EmotionImagesize, EmotionImageSize))
        
        #rgb2gray emotion and age images
        #.....
        #

        GenderValidateImages[0, :, :, :] = Gender_resized_img
        AgeValidateImages[0, :, :] = Age_resized_img
        EmotionValidateImages[0, :, :] = Emotion_resized_img
        
        time4 = time.time()

        # Predict gender
        result = GenderRecognitionSession.run(vgg_rgb.pred, feed_dict={X: ValidateImages, train: False})
        AgeResult = AgeEstimationSession.run( AgePred, feed_dict={AgeInputImage: AgeValidateImages } )
        EmotionResult = EmotionEstimationSession.run( EmotionPred, feed_dict={EmotionInputImage: EmotionValidateImages } )
        
        
        
        
        predict = result[0]

        time5 = time.time()
        
        # print strings
        print ("predict : " ) + str(predict)
        if predict==0:
            text = "Male"
        else:
            text = "Female"

        # Draw rectangle and put text on frame
        cv2.rectangle(frame, (x,y), (x2,y2), (0,255,0),3)
        cv2.putText(frame,text,(x2+10,y2-10), font, 1,(0,0,0),3)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    time6 = time.time()

    print ("=========== running time ===========")
    print ("read camera\t: ") + str(time2-time1)
    print ("face detection\t: ") + str(time3-time2)
    print ("face crop\t: ") + str(time4-time3)
    print ("face prediction\t: ") + str(time5-time4)
    print ("total time\t: ") + str(time6-time1)
    print ("================================\n")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


GenderRecognitionSession.close()
AgeEstimationSession.close()
EmotionEstimationSession.close()



