import base64
import csv
from flask import Flask, jsonify, render_template, Response, request
import os
import cv2
import joblib
import numpy as np
import datetime
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.models import load_model
import tensorflow as tf
import dlib
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
import subprocess

app = Flask("FRAMS", static_folder='templates')

#face detection xml
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initiate the saved model variable
loaded_model = load_model('trainer/my_model_dlib.keras')

# Load the face detector, landmark predictor, and face recognition model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Function to extract facial encodings from an image
# it uses face_detector, shape_predictor, face_encoder method to generate encodings.
def face_encodings(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 1)
    encodings = [np.array(face_encoder.compute_face_descriptor(image, shape_predictor(gray, rect))) for rect in rects]
    return encodings

# URl for start screen
@app.route('/')
def index():
    return render_template('start_screen.html')

# URl for register screen
@app.route('/register_screen')
def register_screen():
    return render_template('register_screen.html')

# URl for attendence screen
@app.route('/attendence_screen')
def attendence_screen():
    return render_template('attendence_screen.html')

# URl to open a web cam
@app.route('/open-webcam')
def open_webcam():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Handle the image and form data
# Called from register screen to send captured image and form data of a student while registering. 
@app.route('/save_student_data', methods=['POST'])
def handle_save_image():
    print("saveimage method", request.json.get("jsonData") )

    json_data = request.json  # Accessing jsonData from the request JSON payload
    if json_data:
        image_data = json_data.get("image_data")  # Accessing image_data from jsonData
        form_data = json_data.get("form_data") # Accessing form_data from jsonData

        #Calling Save_image function , send image data and form data to fetch name and id to label the image data
        save_image(image_data, form_data)

        return 'Image saved successfully!', 200

    else:
        return "Invalid request", 400

# Route to render the table view with data from the CSV file
# Called from start screen
@app.route('/check_attendence')
def check_attendence():
    # Read data from CSV file
    data = []
    with open('csv_files/attendence.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the first row (headers)

        for row in csv_reader:
            data.append(row)
            
    return render_template('attendence_table_view.html', data=data)

# Method to register user after capturing pcitures
# It calls the model file for training on the new face
# Called from register screen, from register button action
@app.route('/register_user', methods=['POST'])
def process_registration():
    try:
        # Execute the FaceRecognitionModel.py file
        subprocess.run(['python', 'models/FaceRecognitionModel.py'])
        return jsonify({"message": "ok"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Load the webcam capture script here
# Called from register and attendence screen for accessing a webcam
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)  # 1 for horizontal flip, 0 for vertical flip, -1 for both

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #save_image(cap)
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


#This method is used to find name of a student from csv file for a student id after being recognised.
def find_names(id):
    print("id in find names",id)
    with open('csv_files/form_data.csv', 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Student Id'] == id:
                name = row['Student Name']
                print("id in find names",name)
                return name
            
# this method saves the registered images with proper labels by detecting the face coordinates, 
# perform some preprocessings, cropping and resizing to (250, 250) dimensions.
def save_image(image_data, form_data):
    image_data = image_data.split(',')[1]  # Remove 'data:image/jpeg;base64,' prefix
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Calling image_cleaning_resizing function to resize and crop the image to get a face
    cropped_image = image_cleaning_resizing(img)

    #Creating label for the image with student name and student id
    id = form_data.get("Student Id") 
    name = form_data.get("Student Name")
    image_id =  id+"_"+name

    #Creating a directory with person's name
    directory = os.path.join('images', id)
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

    # Writing the cropped processed image to a folder
    cv2.imwrite(image_path, cropped_image)

    #Sending a form data further for saving 
    #Calling save_form_data to save the form details
    if(save_form_data(form_data)):
        return 'Form data saved successfully!', 200
    
   
#Function to detect face and crop the image, resize it and return
#Preprocessing method , checks the necessary condition and return the resized face to save to calling fucntion
def image_cleaning_resizing(image):
    if isinstance(image, str):
        # Load the image from file
        image = cv2.imread(image)
       
    elif not isinstance(image, np.ndarray):
        # If image is not a NumPy array, return None
        return None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    # Crop and save each detected face
    #for i, (x, y, w, h) in enumerate(faces):
    if len(faces) > 0:
        #Detecting first face in the image
        x, y, w, h = faces[0]

        # Crop the detected face with a small margin
        margin = 0.2
        x_margin = int(w * margin)
        y_margin = int(h * margin)
        #face = image[y - y_margin:y + h + y_margin, x - x_margin:x + w + x_margin]

        face = image[max(0, y - y_margin):min(y + h + y_margin, image.shape[0]),
                     max(0, x - x_margin):min(x + w + x_margin, image.shape[1])]
        # Resize the cropped face to a standard size
        target_size = (250, 250)
        face_resized = cv2.resize(face, target_size)
        return face_resized
    
# This method saves the attendence of the student, by taking the image, converting into encodings 
# encodings are fed to the model, that predicts the correct face and returns the label value of the image 
# Label is then checked for a name of the student to display
# Called from the attendence screen
@app.route('/save_attendence', methods=['POST'])
def handle_save_attendence():
    name = ""
    json_data = request.json  # Accessing jsonData from the request JSON payload
    if json_data:
        image_data = json_data.get("image_data")  # Accessing image_data from jsonData
        image_data = image_data.split(',')[1]  # Remove 'data:image/jpeg;base64,' prefix
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)) 

        if len(faces) > 0:
            x, y, w, h = faces[0]
            margin = 0.2
            x_margin = int(w * margin)
            y_margin = int(h * margin)
            face = image[y - y_margin:y + h + y_margin, x - x_margin:x + w + x_margin]

            #Incoming picture during attendence is resized and encoded, as the model takes the 128 encoded picture for recognition
            face_resized = cv2.resize(face, (250, 250))
            encface_encs = face_encodings(face_resized)

            # Ensure that encface_encs is not empty
            if len(encface_encs) > 0:
                encface_encs = np.array(encface_encs)  # Convert list to numpy array

                # Predict using the loaded model
                loaded_model = load_model('trainer/my_model_dlib.keras')
                predictions = loaded_model.predict(encface_encs)

                # Initialize LabelEncoder
                le = joblib.load('label_encoder.pkl')

                # Convert predictions to labels
                predicted_labels_encoded = np.argmax(predictions, axis=1)  # Assuming output is one-hot encoded
                predicted_label = le.inverse_transform(predicted_labels_encoded)
        
                idLabel = int(predicted_label[0])  # Access the first (and only) element of the list and convert it to an integer
                print("label value predicted", predicted_label, "extracted", idLabel)

                image_filename = f'{idLabel}.jpg'
                os.makedirs("new_images", exist_ok=True)

                # Construct the full file path
                filepath = os.path.join("new_images", image_filename)
                cv2.imwrite(filepath, face_resized)
                
                # Get the corresponding name for the predicted label to be shown to the user
                name = find_names(str(idLabel))
          
                #response object created to send to the calling fucntion with name and id label for display
                response_data = {
                    'message': 'Image saved successfully!',
                    'name': name,  # Include the name in the response
                    'id': idLabel
                }

                # Return the response as JSON
                return jsonify(response_data), 200
            else:
                print("No face detected")
                return "No face detected", 400
        else:
            print("No face detected")
            return "No face detected", 400
    else:
        print("Invalid request")
        return "Invalid request", 400



#this method is called when user says okay to the alery showing his name and then it saves its attendence to the csv file
#Called from the attendence screen
@app.route('/save_attendence_in_csv', methods=['POST'])
def save_attendence_in_csv():
    print("saving attendence in csv")
    json_data = request.json  # Accessing jsonData payload from the request JSON payload from attendence screen
    if json_data:
        name = json_data.get("name")  # Accessing image_data from jsonData
        id = json_data.get("id")   # Remove 'data:image/jpeg;base64,' prefix

        # Get the current system date and time
        current_datetime = datetime.datetime.now()
        date = current_datetime.strftime("%Y-%m-%d")
        time = current_datetime.strftime("%H:%M:%S")

        # Check if the attendance entry already exists for a particular id and date by calling is_attendence_saved method
        #then it send the message that the attendence is already been marked
        if is_attendence_saved(id, date):
            print("Attendence for", name, "on", date, "is already saved.")
            message = f" Your attendance for {date} has already been marked."
            response_data = {
                'message': message
            }
        # if not marked already, then saves the entry for that student
        else:
            with open('csv_files/attendence.csv', 'a', newline='') as csvfile:
                fieldnames = ['id', 'name', 'date', 'time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header if the file is empty
                if csvfile.tell() == 0:
                    writer.writeheader()

                # Write data
                writer.writerow({'id': id, 'name': name, 'date': date, 'time': time}) #save_image(image_data, form_data)"""

                message = f"Attendance saved successfully!"
                response_data = {
                    'message': message,
                }

        # Return the response as JSON
        return jsonify(response_data), 200
    
    else:
        print("response is 400")
        return "Invalid request", 400

#Method used to check if the attendence is already marked or not by checking the attendence csv file
def is_attendence_saved(id, date):
    try:
        with open('csv_files/attendence.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == id and row[2] == date:
                    return True
                
    except FileNotFoundError:
        return False  # Return False if the file doesn't exist
    return False

#Saving the form data in a csv files
# If the record exists , it will not create a duplicate copy, rather replace the older one
def save_form_data(form_data):
    print("save_form_data")
    csv_file_path = 'csv_files/form_data.csv'
    record_exists = False

    # Read existing records from CSV file
    existing_records = []
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            existing_records.append(row)

    # Check if the record already exists
    for record in existing_records:
        if record['Student Id'] == form_data['Student Id']:
            record_exists = True
            record.update(form_data)
            break

    # Write records back to CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = form_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # Write all existing records except the one being updated
        for record in existing_records:
            if record['Student Id'] != form_data['Student Id']:
                writer.writerow(record)

        # Write the updated record if it exists
        if record_exists:
            writer.writerow(form_data)
        # Write a new record if it doesn't exist
        else:
            writer.writerow(form_data)

    return 'Form data saved successfully!', 200

if __name__ == '__main__':
    app.run(debug=True)
