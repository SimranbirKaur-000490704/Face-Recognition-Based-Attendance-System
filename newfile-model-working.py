import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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

    
    def _preprocess_data(self, images, labels):
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        print("hello",len(labels_encoded))
        self.num_classes = len(le.classes_)
        print("num ckasses",self.num_classes)
        images = np.array(images)
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)  # Add channel dimension
        images = images.astype('float32') / 255  # Normalize pixel values

        print("len of images", len(images))

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

    def train(self, test_size=0.2, epochs=10, batch_size=16):
        images, labels = self._load_images_and_labels()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels)
        
        print("Number of training samples:", len(self.X_train))
        print("Number of training labels:", len(self.y_train))
        print("Number of testing samples:", len(self.X_test))
        print("Number of testing labels:", len(self.y_test))

        self.X_train, self.y_train = self._preprocess_data(self.X_train, self.y_train)
        self.X_test, self.y_test = self._preprocess_data(self.X_test, self.y_test)
       
        self.build_model()
        print(self.model.summary()) 
       # self.model.fit(self.X_train, [self.y_train[:, 0], self.y_train[:, 1]], batch_size=32,
        #  validation_data = (self.X_test, [self.y_test[:, 0], self.y_test[:, 1]] ),
         # epochs=100)
       
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=32, validation_split=0.1)

    def evaluate(self):
        _, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return accuracy

# Example usage:
model = FaceRecognitionCNN(images_folder='images1')
model.train()
accuracy = model.evaluate()
print("Accuracy:", accuracy)
accuracy_percentage = accuracy * 100
print("Accuracy:", accuracy_percentage, "%")
