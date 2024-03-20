from mtcnn import MTCNN
import cv2

import os
def image_cleaning_resizing(image1):

    # Load the image
    image = cv2.imread(image1)

    # Initialize the MTCNN detector
    detector = MTCNN()

    # Detect faces in the image
    faces = detector.detect_faces(image)

    # Initialize a counter for image filenames
    counter = 1

    # Iterate through all detected faces
    for face in faces:
        # Extract the coordinates of the bounding box
        x, y, w, h = face['box']
        
        # Crop the detected face with a small margin
        margin = 0.2
        x_margin = int(w * margin)
        y_margin = int(h * margin)

        print(x_margin)
        print(y_margin)
        cropped_face = image[y - y_margin:y + h + y_margin, x - x_margin:x + w + x_margin]

        # Resize the cropped face to a standard size (250x250)
        resized_face = cv2.resize(cropped_face, (250, 250))

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
        cv2.imwrite(image_path, resized_face)


folder_path ="extra_images"
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, filename)
        # Preprocess and save the image
        image_cleaning_resizing(image_path)