# Face-Recognition-Based-Attendance-System

Introducing Facial Recognition Attendance Management System in Educational Institutions
with - A Deep Learning Approach

This facial recognition attendance management system tailored for educational institutions, boasting an impressive accuracy rate of 99%. The system adopts a dual-model framework, integrating a Dlib-based Convolutional Neural Network (CNN) model for robust facial feature encoding alongside a distinct Feedforward Neural Network (FNN) for recognition purposes. This innovative approach enables precise and swift student identification. Notably, the system effectively addresses the shortcomings inherent in traditional attendance methodologies, including time-intensive processes, error-prone recording, and privacy apprehensions.


In this project:

    1.  app.py - This is the main file to run this project.

    2.  MODEL FILE
        FOLDER - model
        FaceRecognitionModel.py - This is the model file functional for recognising faces

    3.  FOLDER - templates
        This folder contains all the html pages used in this project for various fucntions
        a. start_screen.html: This is the start screen of the application. When app.py is started , this is the first page taht shows up. It has link to other three pages.

        b. Register_screen.html: This page is used to register a new student to this application. First, student has to fill the application form , and submit it with its 10 images. Then after submitting, students images and form saves in the folder. Student has to press the register button, which will train the model on the new face. Once the regitration is completed, it shows the message saying " Registration is completed, use can take the attendence".

        c. attendence_screen.html: This page takes the attendence of the student by clicking the image and it throws the message with the student name, for which user has to confirm its identity and submit it. It saves the attendence to csv file.

        d. attendence_table_view.html: This page displays the saved attendence of the student along with student id and timestamp on the screen.

    4. FOLDER - csv_files
       This folder holds the csv files that are saved in this project.
       a. attendence.csv
       b. form_data.csv
    
    5. FOLDER - dataset/images
       This is the folder that holds the dataset of this project after preprocessing with proper folders inside. Folder name is the student id of the student saved during the registration.

    6. FOLDER - Dlib_files
       This contains the files that are used for encoding the faces and detecting landmarks in the face.

    7. FOLDER - experimented_models
       This folder has all the models that are previously experimented before coming to the final model which is saved in the model folder.

    8. FOLDER -  helper_files
       This contains the helper files that are used in this project - HAAR cascade for face detection and pickle file for label encoding.

    9. FOLDER - trainer
       This folder has the saved model file that is being used in the project for face recognition.
       - face_recognition_model.keras (final model used)
       - cnn_model.keras (experimented model)
       - Lbph_model.yml(experimented model)

    10. requirement.txt
        this file contain all the dependencies used in the project


    

    
    
    

