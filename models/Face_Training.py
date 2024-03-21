import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

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

image_path = 'images'
path = collect_image_paths(image_path)

def getImagesAndLabels(path):

    faceSamples=[]
    ids = []

    for imagePath in path:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
            
        directory, filename = os.path.split(imagePath)
        #id = int(os.path.split(imagePath)[-1].split(".")[1])
       
        id = int(filename.split(imagePath)[0].split('_')[0])

        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    
    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...",)
faces,ids = getImagesAndLabels(path)

recognizer.train(faces, np.array(ids))

directory = 'trainer'
if not os.path.exists(directory):
    os.makedirs(directory)

# Write the trainer data to the file
recognizer.write(os.path.join(directory, 'trainer.yml'))

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))