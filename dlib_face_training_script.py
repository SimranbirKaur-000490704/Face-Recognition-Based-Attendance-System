import os
import numpy as np
import dlib
from skimage import io

# Path for face image database
image_dir = 'images'
trainer_dir = 'trainer'

# Initialize the Dlib face recognition model
face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()

# Function to collect image paths
def collect_image_paths(path):
    image_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Function to extract face embeddings and labels
def extract_embeddings_and_labels(image_paths):
    embeddings = []
    labels = []
    for image_path in image_paths:
        label = os.path.basename(os.path.dirname(image_path))
        img = io.imread(image_path)
        dets = face_detector(img, 1)
        for detection in dets:
            shape = dlib.full_object_detection()
            shape.rect = detection
            face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
            embedding = np.array(face_descriptor)
            embeddings.append(embedding)
            labels.append(label)
    return embeddings, labels

# Collect image paths
image_paths = collect_image_paths(image_dir)

# Extract embeddings and labels
embeddings, labels = extract_embeddings_and_labels(image_paths)

# Convert labels to numeric format if needed

# Save embeddings and labels
np.save(os.path.join(trainer_dir, 'embeddings.npy'), embeddings)
np.save(os.path.join(trainer_dir, 'labels.npy'), labels)

print("Face training completed successfully!")
