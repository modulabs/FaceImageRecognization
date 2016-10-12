from __future__ import print_function
import time
import requests
import cv2
import operator
import numpy as np

# Import library to display results
import matplotlib.pyplot as plt
#%matplotlib inline
# Display images within Jupyter

# Variables

_url = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
_key = '5dfa2ad5562c49a69a1071a80072798a' #Here you have to paste your primary key
_maxNumRetries = 10

def processRequest( json, data, headers, params ):

    """
    Helper function to process the request to Project Oxford

    Parameters:
    json: Used when processing images from its URL. See API Documentation
    data: Used when processing image read from disk. See API Documentation
    headers: Used to pass the key information and the data type request
    """

    retries = 0
    result = None

    while True:

        response = requests.request( 'post', _url, json = json, data = data, headers = headers, params = params )

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
                       color = (255,0,0), thickness = 5 )


    for currFace in result:
        faceRectangle = currFace['faceRectangle']
        currEmotion = max(currFace['scores'].items(), key=operator.itemgetter(1))[0]


        textToWrite = "%s" % ( currEmotion )
        cv2.putText( img, textToWrite, (faceRectangle['left'],faceRectangle['top']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1 )

def Emotion ( pathToFileInDisk ):
    # Load raw image file into memory
    with open( pathToFileInDisk, 'rb' ) as f:
        data = f.read()

    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key
    headers['Content-Type'] = 'application/octet-stream'

    json = None
    params = None

    result = processRequest( json, data, headers, params )

    if result is not None:
        # Load the original image from disk
        data8uint = np.fromstring( data, np.uint8 ) # Convert string to an unsigned int array
        img = cv2.cvtColor( cv2.imdecode( data8uint, cv2.IMREAD_COLOR ), cv2.COLOR_BGR2RGB )

        renderResultOnImage( result, img )

        #ig, ax = plt.subplots(figsize=(15, 20))
        #ax.imshow( img )

    #Emotion
    emotion = result[0]['scores']
    return emotion


#TODO: file path

#pathToFileInDisk = './Taehee/kim_1.jpg'

def emo_final_result ( input_img='./face/imwrite.jpg' ):
    dic_emotion = {'sadness': 0, 'neutral': 1, 'contempt': 2, 'disgust': 3, 'anger': 4, 'surprise': 5, 'fear': 6, 'happiness': 7} 
    max_emo_result = 0
    emo_result = Emotion( input_img )
    print (emo_result)

    for emotion, score in emo_result.iteritems():
        
        if score == max(emo_result.values()):

            max_emo_result = emotion

            #print(emotion)
            return dic_emotion[max_emo_result]