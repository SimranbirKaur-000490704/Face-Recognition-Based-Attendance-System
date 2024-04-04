import joblib
import numpy as np
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
                    img = cv2.resize(img, (250, 250))  # Resize image to 250x250
                    images.append(img)
                    labels.append(os.path.basename(root))  # Label is the folder name
            self.labels_dict[label] = os.path.basename(root)
            label += 1
        return images, labels
    
    def _preprocess_data(self, images, labels):
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        joblib.dump(le, 'helper_files/label_encoder.pkl')

        self.num_classes = len(le.classes_)
        images = np.array(images)
        labels_encoded = to_categorical(labels_encoded, self.num_classes)  # Convert labels to one-hot encoding
        return images, labels_encoded

    def build_model(self):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(250, 250, 3))
        for layer in base_model.layers:
            layer.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),  # Added GlobalAveragePooling2D layer
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        self.model = model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, test_size=0.2, epochs=20, batch_size=10):
        images, labels = self._load_images_and_labels()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels)

        self.X_train, self.y_train = self._preprocess_data(self.X_train, self.y_train)
        self.X_test, self.y_test = self._preprocess_data(self.X_test, self.y_test)
        self.build_model()

        # Define callbacks for model checkpointing and early stopping and saving the model to mobilenet_model.h5
        checkpoint = ModelCheckpoint('trainer/mobilenet_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        self.model.fit(self.X_train, self.y_train, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=batch_size,
                       callbacks=[checkpoint, early_stopping], verbose=1)

    def testing_model(self):
        y_pred_prob = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy:", accuracy)


#usage:
model = FaceRecognitionCNN(images_folder='dataset/images')
model.train()
model.testing_model()