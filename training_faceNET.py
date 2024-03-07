import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from facenet import FaceNet

# Path for face image database
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the pre-trained FaceNet model
facenet_model = FaceNet()

# Function to get the images and label data
def collect_image_paths(path):
    image_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    return image_paths

# Function to get FaceNet embeddings
def get_embeddings(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detectMultiScale(img)
    embeddings = []

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        embedding = facenet_model.get_embedding(face)
        embeddings.append(embedding)

    return embeddings

# Function to get images and labels
def get_images_and_labels(path):
    face_samples = []
    ids = []

    for image_path in path:
        embeddings = get_embeddings(image_path)

        directory, filename = os.path.split(image_path)
        id = int(filename.split('_')[0])

        face_samples.extend(embeddings)
        ids.extend([id] * len(embeddings))

    return face_samples, ids

print("\n [INFO] Training faces. It will take a few seconds. Wait ...",)
faces, ids = get_images_and_labels(collect_image_paths('images'))

# Convert lists to NumPy arrays
faces = np.array(faces)
ids = np.array(ids)

# Define a simple classifier
model = Sequential([
    Flatten(input_shape=(128,)),  # 128 is the size of FaceNet embeddings
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(ids)), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(faces, ids, epochs=10, batch_size=32, verbose=2)  # Adjust epochs and batch_size as needed

# Save the classifier model
directory = 'trainer'
if not os.path.exists(directory):
    os.makedirs(directory)

model.save(os.path.join(directory, 'facenet_classifier.h5'))

# Print the number of faces trained and end the program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))