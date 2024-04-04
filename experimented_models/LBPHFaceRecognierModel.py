###################### LBPHFaceRecognizerModel Class#####################
import cv2
import numpy as np
from PIL import Image
import os

#OpenCV/CV2 has a LBPHFaceRecognizer recognizer model for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()

#HAARCASCADE is used for face detection
cascade_path = 'helper_files/haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade_path)

# function to get the images and label data
def collect_image_paths(path):
    image_paths = []
    # Walk through the directory tree starting from the specified path
    for root, dirs, files in os.walk(path):
        # Iterate over all files in the current directory
        for file in files:
            # Check if the file has a common image extension
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Construct the full path to the image file
                image_path = os.path.join(root, file)
                # Add the image path to the list
                image_paths.append(image_path)
    return image_paths


# Face images are collected with labels
def getImagesAndLabels(path):
    faceSamples=[]
    ids = []
    for imagePath in path:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
            
        directory, filename = os.path.split(imagePath)     

        #id is the label for a picture which is obtained from the filename of the picture  
        id = int(filename.split(imagePath)[0].split('_')[0])

        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids


# this method is used to train the model with picture
def trainLBPHModel():
    #Images folder path
    image_path = 'dataset/images'
    path = collect_image_paths(image_path)

    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...",)
    faces,ids = getImagesAndLabels(path)

    #Training the recogniser
    recognizer.train(faces, np.array(ids))

    directory = 'trainer'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the trainer data to the file
    recognizer.write(os.path.join(directory, 'Lbph_model.yml'))

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

#Calling the trainLBPHModel method to execute the class 
trainLBPHModel()