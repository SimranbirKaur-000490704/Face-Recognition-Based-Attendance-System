import cv2
import numpy as np
import os
#import face_recognition

def image_cleaning_resizing(image):
        # Ensure that the input image is in the correct format
    if isinstance(image, str):
        # Load the image from file

        image = cv2.imread(image)
       
    elif not isinstance(image, np.ndarray):
        # If image is not a NumPy array, return None

        return None
    
    print(image.shape)
    
    # Convert the image to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    #img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #faces = face_recognition.face_locations(img_rgb)[0]

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    # Crop and save each detected face
    if len(faces) > 0:
        x, y, w, h = faces[0]
    #for i, (x, y, w, h) in enumerate(faces):
        # Crop the detected face with a small margin
        margin = 0.2
        x_margin = int(w * margin)
        y_margin = int(h * margin)
        face = image[y - y_margin:y + h + y_margin, x - x_margin:x + w + x_margin]

        # Resize the cropped face to a standard size
        target_size = (250, 250)
        face_resized = cv2.resize(face, target_size)

        #Creating label for the image
        id = "3333335"
        name = "Supreet"
        image_id =  id+"_"+name

        #Creating a directory with person's name
        directory = os.path.join('images_new', id)

        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the cropped image to a folder with the unique ID as the filename
        # if the same person comes in and create multiple pics , it will save the images with a numeric labels appended with it.
        image_filename = f'{image_id}.jpg'
        count = 1
        while os.path.exists(os.path.join(directory, image_filename)):
            image_filename = f'{image_id}_{count}.jpg'
            count += 1
                
        image_path = os.path.join(directory, image_filename)
        cv2.imwrite(image_path, face_resized)

# Iterate over all files in the folder

folder_path ="extra_images"
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, filename)
        # Preprocess and save the image
        image_cleaning_resizing(image_path)