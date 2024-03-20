import joblib
import numpy as np
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#from keras.preprocessing.image import Rescaling, RandomFlip, RandomRotation, RandomZoom
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
#from tensorflow.keras.layers.preprocessing import Rescaling, RandomFlip, RandomRotation, RandomZoom
from sklearn.metrics import accuracy_score

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
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (250, 250))  # Resize image to 100x100
                    #img = cv2.resize(img, (96, 96))  # Resize image to 100x100
                    #img = np.expand_dims(img, axis=-1)  # Add channel dimension

                    images.append(img)

                    # added code to get the student if to create a label
                    #directory, filename = os.path.split(image_path)
                    
                    #label = int(filename.split('_')[0])
                    #print("label",label)

                    #removed this
                    labels.append(os.path.basename(root))  # Label is the folder name
            self.labels_dict[label] = os.path.basename(root)
            #self.labels_dict[label] = label      
            label += 1
        return images, labels
    
    def _preprocess_data(self, images, labels):
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        joblib.dump(le, 'label_encoder.pkl')
        
        print("normal labels", labels)
        print("hello",labels_encoded)

        self.num_classes = len(le.classes_)
        print("num cLasses",self.num_classes)
        images = np.array(images)
        #images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 3)  # Add channel dimension
        #images = images.astype('float32') / 255  # Normalize pixel values

        print("len of images", len(images))
        labels  = np.array(labels)

        labels_encoded = to_categorical(labels_encoded, self.num_classes)  # Convert labels to one-hot encoding       
        return images, labels_encoded
    
    """def build_model(self):
        self.model = Sequential([
            Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', input_shape=(250, 250, 1)),
            Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'),
            MaxPooling2D((2, 2), strides=(2, 2), name='pool1'),

            # Block 2
            Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'),
            Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'),
            MaxPooling2D((2, 2), strides=(2, 2), name='pool2'),

            # Block 3
            Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'),
            Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'),
            Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'),
            MaxPooling2D((2, 2), strides=(2, 2), name='pool3'),

            # Block 4
            Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'),
            Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'),
            Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3'),
            MaxPooling2D((2, 2), strides=(2, 2), name='pool4'),

            # Block 5
            Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1'),
            Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2'),
            Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3'),
            MaxPooling2D((2, 2), strides=(2, 2), name='pool5'),

            # Flatten
            Flatten(name='flatten'),

            # Fully connected layers
            Dense(4096, activation='relu', name='fc6'),
            Dense(4096, activation='relu', name='fc7'),
            Dense(self.num_classes, activation='softmax', name='fc8'),
        ])"""
            # Create model

           # model = Model(inputs=input_layer, outputs=output, name='vggface_vgg16')

    def build_model(self):
        self.model = Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.Rescaling(1./255), # Normalizaing pixel values
            Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (2, 2), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            #Dropout(0.3),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ]) 
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    """def build_model(self):
        self.model = Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.Rescaling(1./255), # Normalizaing pixel values
            Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            #Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            #Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(512, (3, 3), activation='relu'),
            #Conv2D(512, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            #Dense(512, activation='relu'),
           # Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])  """                                            
        

    def train(self, test_size=0.2, epochs=20, batch_size=10):
        images, labels = self._load_images_and_labels()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels)
        
        # Set X and y attributes
        self.X = images
        self.y = labels
        
        print("Number of training samples:", len(self.X_train))
        print("Number of training labels:", len(self.y_train))
        print("Number of testing samples:", len(self.X_test))
        print("Number of testing labels:", len(self.y_test))

        self.X_train, self.y_train = self._preprocess_data(self.X_train, self.y_train)
        self.X_test, self.y_test = self._preprocess_data(self.X_test, self.y_test)
        self.build_model()
       # print(self.model.summary()) 
        
       # checkpointer = ModelCheckpoint(filepath='my_model.keras', verbose=1, save_best_only=True)
        epochs = 20
        #self.model.fit(self.X_train, self.y_train, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

                
        # Train the model
        self.model.fit(self.X_train, self.y_train, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, verbose=1)

        # Save the trained model
        self.model.save('my_model.keras')


    def testing_model(self):
        
        # Test the model on the test set
        y_pred_prob = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Convert predicted probabilities to class labels
        print(y_pred)
        y_true = np.argmax(self.y_test, axis=1) 

        print("pred",y_pred)
        print("true",y_true)

        accuracy = accuracy_score(y_true, y_pred)
        print("accuracy", accuracy)


# Example usage:
model = FaceRecognitionCNN(images_folder='images_new')
model.train()
model.testing_model()


