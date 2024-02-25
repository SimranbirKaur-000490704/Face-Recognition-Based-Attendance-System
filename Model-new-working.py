import cv2
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FaceRecognitionTrainer:
    def __init__(self, recognizer, detector, image_path):
        self.recognizer = recognizer
        self.detector = detector
        self.image_path = image_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.accuracy = None

    def collect_image_paths(self):
        image_paths = []
        # Walk through the directory tree starting from the specified path
        for root, dirs, files in os.walk(self.image_path):
            # Iterate over all files in the current directory
            for file in files:
                # Check if the file has a common image extension
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    # Construct the full path to the image file
                    image_path = os.path.join(root, file)
                    # Add the image path to the list
                    image_paths.append(image_path)
        return image_paths

    def get_images_and_labels(self, path):
        face_samples = []
        ids = []

        for image_path in path:
            PIL_img = Image.open(image_path).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            
            directory, filename = os.path.split(image_path)
            id = int(filename.split('_')[0])

            faces = self.detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

        return face_samples, ids

    def train(self, test_size=0.2, epochs=10, batch_size=2):
        # Collect image paths
        image_paths = self.collect_image_paths()

        # Get images and labels
        faces, ids = self.get_images_and_labels(image_paths)

        # Split dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            faces, ids, test_size=0.2, random_state=42
        )

        # Train recognizer
        self.recognizer.train(self.X_train, np.array(self.y_train))

        # Predict labels for testing set
        y_pred = [self.recognizer.predict(face)[0] for face in self.X_test]

        # Compute accuracy
        self.accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", self.accuracy*100)
    
        # Write the trained model to a file
        directory = 'trainer'
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.recognizer.write(os.path.join(directory, 'trainer.yml'))

        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

# Example usage:
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image_path = 'images'

trainer = FaceRecognitionTrainer(recognizer, detector, image_path)
trainer.train()
