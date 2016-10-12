from __future__ import print_function
import os
import time
import requests
import cv2
import operator
import numpy as np

# Import library to display results
import matplotlib.pyplot as plt
# Display images within Jupyter
import httplib, urllib, base64 #General API Usage

# Variables

_url_detect = 'https://api.projectoxford.ai/face/v1.0/detect'
_url_verify = 'https://api.projectoxford.ai/face/v1.0/verify'
_url_group = 'https://api.projectoxford.ai/face/v1.0/group'
_key = '0b8afa8909154432bafeed6a3217fcde' #Here you have to paste your primary key
_maxNumRetries = 10


def detectionRequest( json, data, headers, params ):

    """
    Helper function to process the request to Project Oxford

    Parameters:
    json: Used when processing images from its URL. See API Documentation
    data: Used when processing image read from disk. See API Documentation
    headers: Used to pass the key information and the data type request
    retries = 0
    """

    result = None

    while True:

        response = requests.request( 'post', _url_detect, json = json, data = data, headers = headers, params = params )

        if response.status_code == 429:

            print( "Message: %s" % ( response.json()['error']['message'] ) )

            if retries <= _maxNumRetries:
                time.sleep(1)
                retries += 1
                continue
            else:
                print( 'Error: failed after retrying!' )
                break

        elif response.status_code == 200 or response.status_code == 201:

            if 'content-length' in response.headers and int(response.headers['content-length']) == 0:
                result = None
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str):
                if 'application/json' in response.headers['content-type'].lower():
                    result = response.json() if response.content else None
                elif 'image' in response.headers['content-type'].lower():
                    result = response.content
        else:
            print( "Error code: %d" % ( response.status_code ) )
            print( "Message: %s" % ( response.json()['error']['message'] ) )

        break

    return result

def renderResultOnImage( result, img ):

    """Display the obtained results onto the input image"""

    for currFace in result:
        faceRectangle = currFace['faceRectangle']
        cv2.rectangle( img,(faceRectangle['left'],faceRectangle['top']),
                           (faceRectangle['left']+faceRectangle['width'], faceRectangle['top'] + faceRectangle['height']),
                       color = (255,0,0), thickness = 1 )

        faceLandmarks = currFace['faceLandmarks']

        for _, currLandmark in faceLandmarks.items():
            cv2.circle( img, (int(currLandmark['x']),int(currLandmark['y'])), color = (0,255,0), thickness= -1, radius = 1 )

    for currFace in result:
        faceRectangle = currFace['faceRectangle']
        faceAttributes = currFace['faceAttributes']

        textToWrite = "%c (%d)" % ( 'M' if faceAttributes['gender']=='male' else 'F', faceAttributes['age'] )
        cv2.putText( img, textToWrite, (faceRectangle['left'],faceRectangle['top']-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1 )



def run():

    # Face detection parameters
    params = { 'returnFaceAttributes': 'age,gender',
               'returnFaceLandmarks': 'true'}

    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key
    headers['Content-Type'] = 'application/octet-stream'

    json = None

    #cur_dir = '/Users/robot/git_ryan/MS_Face_API/image/Park/park_etc/'
    cur_dir = '/home/modu/DemoDay_ImageProcessing_Team/gwansang/face/'
    test_file = os.listdir(cur_dir)
    #test_file = './face/imwrite.jpg'
    print(cur_dir)
    print(test_file)

    test_faceId = [] #face id info
    file_name = [] #directory file name
    #data_name = [] #data directory

    for i in xrange(0, len(test_file)):
        file_name.append(cur_dir + test_file[i])

        with open (file_name[i], 'rb') as f:
            data = f.read()
            #data_name.append(data)

        try:
            result = detectionRequest( json, data, headers, params )
            #test_faceId.append( result[0]['faceId'])
            test_faceId.append( result[0]["faceAttributes"])
            #time.sleep(1)
            print("success")
        except:
            print("fail filename: %s" %file_name[i])

    #import pdb;pdb.set_trace()
    age_result = test_faceId[0]['age']
    gender_result = test_faceId[0][u'gender']
    #print(age_result, gender)

    return age_result, gender_result
 