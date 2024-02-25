import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
class FaceRecognitionCNN:
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.labels_dict = {}
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.num_classes = None

    def _load_images_and_labels(self):
        images = []
        labels = []
        label = 0
        for root, dirs, files in os.walk(self.images_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_path = os.path.join(root, file)
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (100, 100))  # Resize image to 100x100
                    images.append(img)
                    labels.append(os.path.basename(root))  # Label is the folder name
            self.labels_dict[label] = os.path.basename(root)
            label += 1
        return images, labels


    
    
    #faces,ids = getImagesAndLabels(path)

    def _preprocess_data(self, images, labels):
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        self.num_classes = len(le.classes_)
        images = np.array(images)
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)  # Add channel dimension
        images = images.astype('float32') / 255  # Normalize pixel values
        labels_encoded = to_categorical(labels_encoded, self.num_classes)  # Convert labels to one-hot encoding
        return images, labels_encoded

    def build_model(self):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def collect_image_paths(self, path):
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

    def _getImagesAndLabels(self, path):

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


    #image_path = 'images'
    #new_path = collect_image_paths(image_path)

    def train(self, test_size=0.2, epochs=10, batch_size=2):
        images, labels = self._getImagesAndLabels(self.collect_image_paths('images'))
        print("Labels:", labels)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels)

        # Check if any class has only one member in the test set
        test_labels, test_counts = np.unique(self.y_test, return_counts=True)
        single_member_classes = test_labels[test_counts == 1]
        if len(single_member_classes) > 0:
            print("Single member classes in test set:", single_member_classes)
            # Adjust test_size or remove stratification
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                images, labels, test_size=test_size, random_state=42)  # Remove stratification

        self.X_train, self.y_train = self._preprocess_data(self.X_train, self.y_train)
        self.X_test, self.y_test = self._preprocess_data(self.X_test, self.y_test)
        self.build_model()
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

        def evaluate(self):
            _, accuracy = self.model.evaluate(self.X_test, self.y_test)
            return accuracy

# Example usage:
model = FaceRecognitionCNN(images_folder='images')
model.train()
accuracy = model.evaluate()
print("Accuracy:", accuracy)
