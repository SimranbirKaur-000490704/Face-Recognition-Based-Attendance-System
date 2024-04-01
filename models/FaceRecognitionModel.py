import numpy as np
import os
import cv2
import dlib
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def load_face_recognition_models():
    """
    Load the face detection, landmark predictor, and face recognition models.
    """
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    return face_detector, shape_predictor, face_encoder


def extract_face_encodings(image, face_detector, shape_predictor, face_encoder):
    """
    Extract facial encodings from an image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 1)
    encodings = [np.array(face_encoder.compute_face_descriptor(image, shape_predictor(gray, rect))) for rect in rects]
    return encodings


def preprocess_images_and_labels(images_folder, face_detector, shape_predictor, face_encoder):
    """
    Preprocess images and extract encodings and labels.
    """
    encodings = []
    labels = []
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image is None:
                    logging.error(f"Failed to load image: {image_path}")
                    continue
                face_encs = extract_face_encodings(image, face_detector, shape_predictor, face_encoder)
                encodings.extend(face_encs)
                labels.extend([os.path.basename(root)] * len(face_encs))
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    joblib.dump(le, 'label_encoder.pkl')
    num_classes = len(le.classes_)
    labels_encoded = to_categorical(labels_encoded, num_classes)  # Convert labels to one-hot encoding
    return np.array(encodings), np.array(labels_encoded), num_classes


def build_model(input_shape, num_classes):
    """
    Build the CNN model.
    """
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train the CNN model.
    """
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained CNN model.
    """
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    confidence_scores = np.max(y_pred_prob, axis=1)
    logging.info("Confidence Scores: %s", confidence_scores)
    accuracy = accuracy_score(y_true, y_pred)
    logging.info("Test Accuracy: %s", accuracy)


def loadModel():
    # Load face recognition models
    face_detector, shape_predictor, face_encoder = load_face_recognition_models()

    # Preprocess images and labels
    encodings, labels, num_classes = preprocess_images_and_labels('img', face_detector, shape_predictor, face_encoder)

    # Preprocess data
    X_train, X_test, y_train, y_test = train_test_split(encodings, labels, test_size=0.2, random_state=42,
                                                        stratify=labels)

    # Build CNN model
    model = build_model(input_shape=X_train.shape[1:], num_classes=num_classes)

    # Train CNN model
    trained_model = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate CNN model
    evaluate_model(trained_model, X_test, y_test)



loadModel()
