#-*- coding: utf-8 -*-
from __future__ import division



import pylab as plt
import numpy as np
import tensorflow as tf
import cv2
import time
import os

import sys


# modulabs setting
sys.path.insert(0, "/home/modu/DemoDay_ImageProcessing_Team/gwansang/GenderRecognition")
sys.path.insert(0, "/home/modu/DemoDay_ImageProcessing_Team/gwansang/AgeEstimation/")
sys.path.insert(0, "/home/modu/DemoDay_ImageProcessing_Team/gwansang/EmotionEstimation/")
sys.path.insert(0, "/home/modu/DemoDay_ImageProcessing_Team/gwansang/etc_source/")                


import VGG_RGB
import AgeEstimation_prediction
import emotion_cnn
import return_maxloc
import ReplyCollector

import Age_detect_ms
import Emotion_detect_ms

def max_return(one_hot):
    try:
        max_loc = np.argmax(one_hot, axis = 0)
        return max_loc 
    except:
        print "Something wrong with your array"

print "------------------------------------------------


# In[2]:

batchSize = 1
GenderImageSize = 224
AgeImageSize = 48
EmotionImageSize = 48

epochNum = 0
num_class = 2
learning_rate = 1e-6


# In[3]:

font = cv2.FONT_HERSHEY_SIMPLEX

directoryName = "face data/large"

# check points
GenderCheckPoint_dir = "GenderRecognition/"
AgeCheckPoint_dir = "AgeEstimation/"
EmotionCheckPoint_dir = "EmotionEstimation/"

# ### Gender Recognition Model

# In[4]:

# Create Image space
GenderValidateImages = np.zeros([1, GenderImageSize, GenderImageSize, 3], dtype=np.float32)

# Create Gender Class
vgg_rgb = VGG_RGB.VGG_RGB()


# GenderRecognision input arguments
GenderImage = tf.placeholder(tf.float32, [batchSize, GenderImageSize, GenderImageSize, 3])
GenderLabel = tf.placeholder(tf.int32, [batchSize, 1])
Gender_is_train = tf.placeholder(tf.bool)


# building each model 
GenderRecognitionSession = tf.Session()
vgg_rgb.build(GenderImage, GenderLabel, Gender_is_train, scope="VGG_RGB",  num_classes=num_class, debug=True)

# load model
GenderRecognitionSaver = tf.train.Saver()
Genderckpt = tf.train.get_checkpoint_state(GenderCheckPoint_dir)

if Genderckpt and Genderckpt.model_checkpoint_path:
    GenderRecognitionSaver.restore(GenderRecognitionSession, Genderckpt.model_checkpoint_path)
    print ('load Gender model') 
    
else:
    print ('load fail')

    
    
    
    
    

# ### Emotion Estimation Model


# Create image space
EmotionValidateImages = np.zeros([1, EmotionImageSize, EmotionImageSize], dtype=np.float32)


#EmotionEstimation input arguments
EmotionInputImage = tf.placeholder( tf.float32, [batchSize, EmotionImageSize, EmotionImageSize ] )
keepratio = tf.placeholder(tf.float32)


# building model 
EmotionEstimationSession = tf.Session()
EmotionPred = emotion_cnn.conv_basic( EmotionInputImage )

# load model
EmotionEstimationSaver = tf.train.Saver(emotion_cnn.weight_list)

Emotionckpt = tf.train.get_checkpoint_state(EmotionCheckPoint_dir)

if Emotionckpt and Emotionckpt.model_checkpoint_path:
    EmotionEstimationSaver.restore(EmotionEstimationSession, Emotionckpt.model_checkpoint_path)
    print ('load emotion model' )


    
    
    
    
# create image space 
AgeValidateImages = np.zeros([1, AgeImageSize, AgeImageSize, 1], dtype=np.float32)

AgeInputImage = tf.placeholder(tf.float32, [batchSize, AgeImageSize, AgeImageSize, 1] )


# building model
AgeEstimationSession = tf.Session()
AgePred = AgeEstimation_prediction.model( AgeInputImage )

# load model
print AgeEstimation_prediction.weight_list
AgeEstimationSaver = tf.train.Saver(AgeEstimation_prediction.weight_list)


Ageckpt = tf.train.get_checkpoint_state(AgeCheckPoint_dir)

if Ageckpt and Ageckpt.model_checkpoint_path:
    AgeEstimationSaver.restore(AgeEstimationSession, Ageckpt.model_checkpoint_path)
    print ('load Age model' )




print "camera start =========================="

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

newXSize = 100
newYSize = 100

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    # height, width, channels = frame.shape
    recframe = frame.copy()
    roi_color = np.zeros((1, 1, 3), np.uint8)
    img_height = recframe.shape[0]
    img_width = recframe.shape[1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    key = cv2.waitKey(1)
    
    for (x, y, w, h) in faces:
        if(w>img_width/3):
            break
            

        cv2.rectangle(recframe, (x, y), (x + w, y + h), (255, 0, 0), 3)
        crop_img_gray = gray[y:y + h, x:x + w]
        crop_img = frame[y:y + h, x:x + w]
        

        # Capture
        if key > 0:
           
            cv2.imshow('frame',recframe)    
            cv2.imshow("roi", crop_img)
            cv2.imwrite("face/imwrite.jpg", frame)
            #resize_Image = cv2.resize(crop_img,(newXSize, newYSize))

            # resizing images
            Gender_resized_img = cv2.resize(crop_img, (GenderImageSize, GenderImageSize)) 
            Age_resized_img = cv2.resize(crop_img, (AgeImageSize, AgeImageSize))
            Emotion_resized_img = cv2.resize(crop_img_gray, (EmotionImageSize, EmotionImageSize))


            
            GenderValidateImages[0, :, :, :] = Gender_resized_img
            AgeValidateImages[0, :, :, 0] = Age_resized_img[:, :, 0]
            EmotionValidateImages[0, :, :] = Emotion_resized_img

                        # Predict gender
            result = GenderRecognitionSession.run(vgg_rgb.score_fr, feed_dict={GenderImage: GenderValidateImages, Gender_is_train: False})
            AgeResult = AgeEstimationSession.run( AgePred, feed_dict={AgeInputImage: AgeValidateImages } )
            EmotionResult = EmotionEstimationSession.run( EmotionPred, feed_dict={EmotionInputImage: EmotionValidateImages } )

            
            MyGen = return_maxloc.max_return(result[0])
            MyAge = return_maxloc.max_return(AgeResult[0])
            MyEmo = return_maxloc.max_return(EmotionResult[0])


            MyAge, MyGenS = Age_detect_ms.run()
            MyAge += 1
            MyEmo = Emotion_detect_ms.emo_final_result()
            print "MyEmo : ", MyEmo

            MyAgeRage = 0

            GenString = ""
            AgeString = ""
            EmoString = ""
            MentString = ""
            HelloString = ""
            
            if MyGenS == u'male':
                MyGen = 0
                GenString += "Gender : Male"
            else: #if 1:
                MyGen = 1
                GenString += "Gender : Female"

            if MyAge >= 0 and MyAge <= 5:
                AgeString += str(MyAge)
                MyAgeRage = 0
                if MyGen == 0:
                    HelloString += "Hi, cute boy~  "
                else:
                    HelloString += "Hi, cute girl~  "
                    
            elif MyAge > 5 and MyAge <= 10:
                AgeString += str(MyAge)
                MyAgeRage = 1
                if MyGen == 0:
                    HelloString += "Hi, cute boy~  "
                else:
                    HelloString += "Hi, cute girl~  "
                    
            elif MyAge > 10 and MyAge <= 16:
                AgeString += str(MyAge)
                MyAgeRage = 2
                if MyGen == 0:
                    HelloString += "Hello little boy  "
                else:
                    HelloString += "Hello little girl  "

            elif MyAge > 16 and MyAge <= 28:
                AgeString += str(MyAge)
                MyAgeRage = 3
                if MyGen == 0:
                    HelloString += "Hey, What's up Bro.  "
                else:
                    HelloString += "Hey, Beauty. "

            elif MyAge > 28 and MyAge <= 51:
                AgeString += str(MyAge)
                MyAgeRage = 4
                if MyGen == 0:
                    HelloString += "Welcome, Mr.  "
                else:
                    HelloString += "Welcome, Lady  "
            elif MyAge > 51 and MyAge <= 75:
                AgeString += str(MyAge)
                MyAgeRage = 5
                if MyGen == 0:
                    HelloString += "Good Day, Sir "
                else:
                    HelloString += "How's it going Miss.  "
            elif MyAge > 75:
                AgeString += str(MyAge) 
                MyAgeRage = 6
                if MyGen == 0:
                    HelloString += "Greeting, Sir.  "
                else:
                    HelloString += "I'm Glad to meet you, ma'am  "

 
            if MyAge == 0:
                AgeString += "Age : 1"
                if MyGen == 0:
                    HelloString += "Hi, cute boy~  "
                else:
                    HelloString += "Hi, cute girl~  "
                    
            elif MyAge == 1:
                AgeString += "Age : 5"
                if MyGen == 0:
                    HelloString += "Hi, cute boy~  "
                else:
                    HelloString += "Hi, cute girl~  "
                    
            elif MyAge == 2:
                AgeString += "Age : 10"
                if MyGen == 0:
                    HelloString += "Hello little boy  "
                else:
                    HelloString += "Hello little girl  "

            elif MyAge == 3:
                AgeString += "Age : 16"
                if MyGen == 0:
                    HelloString += "Hey, What's up Bro.  "
                else:
                    HelloString += "Hey, Beauty. "

            elif MyAge == 4:
                AgeString += "Age : 28"
                if MyGen == 0:
                    HelloString += "Welcome, Mr.  "
                else:
                    HelloString += "Welcome, Lady  "
            elif MyAge == 5:
                AgeString += "Age : 51"
                if MyGen == 0:
                    HelloString += "Good Day, Sir "
                else:
                    HelloString += "How's it going Miss.  "
            elif MyAge == 6:
                AgeString += "Age : 75" 
                if MyGen == 0:
                    HelloString += "Greeting, Sir.  "
                else:
                    HelloString += "I'm Glad to meet you, ma'am  "

                
            if MyEmo == 4:
                EmoString += "Emotion : Angry"
                HelloString += "What makes you Angry"
            elif MyEmo == 2:
                EmoString += "Emotion : Contempt"
                HelloString += "Don't hate youself"
            elif MyEmo == 3:
                EmoString += "Emotion : Disgust"
                HelloString += "What's the matter with you"
            elif MyEmo == 6:
                EmoString += "Emotion : Fear"
                HelloString += "Why?? Why? What did you see!?"
            elif MyEmo == 7:
                EmoString += "Emotion : Happy"
                HelloString += "I like your smile"            
            elif MyEmo == 1:
                EmoString += "Emotion : Neutral"
                HelloString += "Hello? can you see me?"
            elif MyEmo == 0:
                EmoString += "Emotion : Sadness"
                HelloString += "That's OK. Don't be upset"           
            elif MyEmo == 5:
                EmoString += "Emotion : Surprise"
                HelloString += "Surprise!!!!"

            MentString += ReplyCollector.makeFortunesByPhysiognomy( MyGen, MyAgeRage, MyEmo )
                
            print ""
            print ""
            print "Code: ", MyGen, " ", MyAge, " ", MyEmo            
            print GenString
            print AgeString
            print EmoString
            print "------------------------------ 당신을 위한 한마디 ------------------------------"
            print ""
            print MentString
            print ""
            print "--------------------------------------------------------------------------------"
            print ""
            print ""
            
            REC_POS = -120
            LINE_GAP = 20
            TOP_LINE = -95
            
            cv2.rectangle(recframe, (0, img_height+REC_POS), (img_width, img_height), (255, 255, 255), -1)
            cv2.putText(recframe,GenString,(10,           img_height+TOP_LINE      ), font, 0.5,(0,0,0),1)
            cv2.putText(recframe,AgeString,(10,            img_height+TOP_LINE+LINE_GAP        ), font, 0.5,(0,0,0),1)
            cv2.putText(recframe,EmoString,(10,img_height+TOP_LINE+2*LINE_GAP), font, 0.5,(0,0,0),1)
            cv2.putText(recframe,HelloString,(10,img_height+TOP_LINE+4*LINE_GAP), font, 0.7,(0,150,50),1)
            cv2.imshow('frame', recframe )
            
            #os.system("python neural-style/neural_style.py --content face/imwrite.jpg --styles neural-style/examples/1-style.jpg --output face/style-image.jpg & ")

            cv2.waitKey(0)

            break
            
    # Display the resulting frame
    cv2.imshow('frame',recframe)
    
    # EXIT
    if key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


GenderRecognitionSession.close()
AgeEstimationSession.close()
EmotionEstimationSession.close()


print "All Session Closed"





