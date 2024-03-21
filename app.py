import base64
import csv
from flask import Flask, jsonify, render_template, Response, request
import os
import cv2
import joblib
import numpy as np
import datetime

#import camera
#print(cv2._version_)
#from tensorflow.keras.models import load_model
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask("FRAMS")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read('trainer/trainer.yml')
font = cv2.FONT_HERSHEY_SIMPLEX

loaded_model = load_model('my_model.keras')

# Load the saved model
#loaded_model = load_model('FR_Model.h5')

@app.route('/')
def index():
    return render_template('start_screen.html')

@app.route('/register_screen')
def register_screen():
    return render_template('register_screen.html')

@app.route('/attendence_screen')
def attendence_screen():
    return render_template('attendence_screen.html')

@app.route('/open-webcam')
def open_webcam():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/open-webcam1')
def open_webcam1():
    return Response(generate_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to render the table view with data from the CSV file
@app.route('/check_attendence')
def check_attendence():
    # Read data from CSV file
    data = []
    with open('attendence.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the first row (headers)

        for row in csv_reader:
            data.append(row)

    
    return render_template('attendence_table_view.html', data=data)


# Load your webcam capture script here
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

def generate_frames1():
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


"""def generate_frames1():
 
    print("here it is")
    faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cap.set(3,500) # set Width
    cap.set(4,500) # set Height
    while True:
        ret, img = cap.read()
        #img = cv2.flip(img, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        )
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]  
        cv2.imshow('video',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()"""

"""def generate_frames1():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Check if the frame is read successfully
            if not success:
                print("Error: Unable to read frame from video capture")
                break


            faces = face_cascade.detectMultiScale(
              gray,     
              scaleFactor=1.2,
              minNeighbors=5,     
              minSize=(20, 20)
            )
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]  

            

            cv2.imshow('video',img)
        
            # Detect faces in the frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h),  (0, 110, 0), 2)
                
                face = img[y:y+h, x:x+w]

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', face)
            face = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + face + b'\r\n')
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #save_image(cap)
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()"""
    
# Load webcam capture script here : original code 
"""def generate_frames1():
    # Load the pre-trained Haar Cascade classifier for face detection    
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # change 6-mar
            
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            

            # Draw rectangles around the detected faces / commented out when using differnet model other than recogniser
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 110, 0), 2)
                 
                margin = 0.2
                x_margin = int(w * margin)
                y_margin = int(h * margin)
                face_gray = gray[y - y_margin:y + h + y_margin, x - x_margin:x + w + x_margin]

                # Resize the cropped face to a standard size
                target_size = (250, 250)
                face_resized = cv2.resize(face_gray, target_size)
                
                # Add channel dimension
                face_batch = np.expand_dims(face_resized, axis=2)  # Add channel dimension
                
                # Normalize pixel values
                face_batch = face_batch / 255.0

                yield face_batch

                #face_batch = np.expand_dims(face_resized, axis=0)
                #face_batch = np.expand_dims(face_batch, axis=3)  # Add channel dimension

                # Initialize LabelEncoder
                le = joblib.load('label_encoder.pkl')
                # Make prediction using the loaded model
                predictions = loaded_model.predict(face_batch)
                
                print("predictions",predictions)
                # Convert prediction to label (or name)
                predicted_label_encoded = np.argmax(predictions)  # Assuming output is one-hot encoded
                print("encoded",predicted_label_encoded)
                predicted_label = le.inverse_transform([predicted_label_encoded])[0]

                print("label value predicted", predicted_label)
                name = find_names(str(predicted_label))

            # Get the corresponding name for the predicted label

                cv2.putText(frame, name, (x+5,y-5), font, 1, (255,255,255), 2)
                #cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

            
            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()"""

#This method is used to find name of student from csv file for a student id
def find_names(id):
    print("id in find names",id)
    with open('csv_files/form_data.csv', 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Student Id'] == id:
                name = row['Student Name']
                print("id in find names",name)
                #names.append(name)
    
                return name
            
# Open the webcam window
"""def open_webcam1():

    vid = cv2.VideoCapture(0)

    while True:
    # Capture the video frame by frame
        ret, frame = vid.read()

    # Check if the frame is not None and has valid dimensions
        if ret and frame.shape[0] > 0 and frame.shape[1] > 0:
        # Display the resulting frame
            cv2.imshow('frame', frame)
            #cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    # the 'q' button is set as the quitting button you may use any desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop release the cap object
    vid.release()

# Destroy all the windows
    cv2.destroyAllWindows()"""


# Saving the image in folder
@app.route('/save_image', methods=['POST'])
def handle_save_image():
    print("saveimage method", request.json.get("jsonData") )

    json_data = request.json  # Accessing jsonData from the request JSON payload
    if json_data:
        image_data = json_data.get("image_data")  # Accessing image_data from jsonData
        form_data = json_data.get("form_data") 

        #Calling Save_image function , send image data and form data to fetch name and id to label the image data
        save_image(image_data, form_data)

        return 'Image saved successfully!', 200

    else:
        return "Invalid request", 400

def save_image(image_data, form_data):
    print("in save images method")
    image_data = image_data.split(',')[1]  # Remove 'data:image/jpeg;base64,' prefix
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Calling image_cleaning_resizing function to resize and crop the image to get a face
    cropped_image = image_cleaning_resizing(img)
    
    print(" image")

    # Check the shape of the image
    if len(cropped_image.shape) == 3:
        # Image has 3 dimensions, indicating it's RGB
        print("RGB image")
    elif len(cropped_image.shape) == 2:
        # Image has 2 dimensions, indicating it's grayscale
        print("Grayscale image")

    #Calling generate unique id function
    #image_id = generate_unique_id()

    #Creating label for the image
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
    cv2.imwrite(image_path, cropped_image)

    save_label_to_csv(image_path, id)

    print(f"Image saved to: {image_path}")

    #Calling save_form_data to save the form details
    save_form_data(form_data)

    #image_path = os.path.join(directory, f'{image_id}.jpg')
    #cv2.imwrite(image_path, cropped_image)
    # Save the cropped image to a folder with the unique ID as the filename
    # image_path = os.path.join('images', f'{image_id}.jpg')
    #cv2.imwrite(image_path, cropped_image)

# Load image paths and corresponding labels from CSV file
"""def load_data(csv_file):
    X = []
    y = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_path, label = row
            # Read image and preprocess it
            image = cv2.imread(image_path)
            #image = preprocess_image(image)  # Implement preprocess_image function as needed
            X.append(image)
            y.append(label)
    return np.array(X), np.array(y)"""


# Function to save the image path and label to a CSV file
def save_label_to_csv(image_path, student_id):
    csv_file = 'csv_files/labels.csv'
    with open(csv_file, 'a') as file:
        file.write(f"{image_path},{student_id}\n")


#Function to detect face and crop the image, resize it and return
def image_cleaning_resizing(image):
       
    if isinstance(image, str):
        # Load the image from file
        image = cv2.imread(image)
       
    elif not isinstance(image, np.ndarray):
        # If image is not a NumPy array, return None
        return None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
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
        face = image[y - y_margin:y + h + y_margin, x - x_margin:x + w + x_margin]

        # Resize the cropped face to a standard size
        target_size = (250, 250)
        face_resized = cv2.resize(face, target_size)

        return face_resized
    
    # Crop and return the first detected face(this logic can be used instaed of the above logic)
    """if len(faces) > 0:
        x, y, w, h = faces[0]
        # Crop the detected face with a small margin
        margin = 0.2
        x_margin = int(w * margin)
        y_margin = int(h * margin)
        face = image[max(0, y - y_margin):min(y + h + y_margin, image.shape[0]),
                     max(0, x - x_margin):min(x + w + x_margin, image.shape[1])]
        # Resize the cropped face to a standard size
        target_size = (250, 250)
        face_resized = cv2.resize(face, target_size)
        return face_resized
    else:
        return None"""
    

# Saving the image in folder : this is a working code /for recognizer model
"""@app.route('/save_attendence', methods=['POST'])
def handle_save_attendence():
    print("handle_save_attendence method", request.json.get("jsonData") )

    json_data = request.json  # Accessing jsonData from the request JSON payload
    if json_data:
        image_data = json_data.get("image_data")  # Accessing image_data from jsonData
        image_data = image_data.split(',')[1]  # Remove 'data:image/jpeg;base64,' prefix
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 110, 0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                #name = find_names(str(id))
                confidence = "  {0}%".format(round(100 - confidence))
                name = find_names(str(id))
            
            else:
               # name = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))


        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Print the current date and time
        print("Current date and time:", current_datetime)

        date = current_datetime.strftime("%Y-%m-%d")
        time = current_datetime.strftime("%H:%M:%S")
        # Write data to CSV file

        with open('attendence.csv', 'a', newline='') as csvfile:
            fieldnames = ['id', 'name', 'date', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if the file is empty
            if csvfile.tell() == 0:
                writer.writeheader()

            # Write data
            writer.writerow({'id': id, 'name': name, 'date': date, 'time': time})


        #Calling Save_image function , send image data and form data to fetch name and id to label the image data
        #save_image(image_data, form_data)

        #Calling save_form_data to save the form details
        #save_form_data(form_data)

        return 'Image saved successfully!', 200

    else:
        return "Invalid request", 400"""


@app.route('/save_attendence', methods=['POST'])
def handle_save_attendence():

    #remove this code, this taking the hardcoded image
    folder_path ="extra_images"
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Construct the full path to the image file
            new_img = os.path.join(folder_path, filename)
            # Preprocess and save the image
            
    #/////-------remove 
    name = ""
    json_data = request.json  # Accessing jsonData from the request JSON payload
    if json_data:
        print("json data not none")
        image_data = json_data.get("image_data")  # Accessing image_data from jsonData
        image_data = image_data.split(',')[1]  # Remove 'data:image/jpeg;base64,' prefix
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        #gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

        #remove
        #image = cv2.imread(new_img)
        #   gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Check the shape of the image
        if len(gray.shape) == 3:
            # Image has 3 dimensions, indicating it's RGB
            print("RGB image")
        elif len(gray.shape) == 2:
            # Image has 2 dimensions, indicating it's grayscale
            print("Grayscale image")
        # Detect faces in the frame
            
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)) 

        # Draw rectangles around the detected faces
        if len(faces) > 0:
            #Detecting first face in the image
            x, y, w, h = faces[0]
            # Crop the detected face with a small margin
            margin = 0.2
            x_margin = int(w * margin)
            y_margin = int(h * margin)
            face_gray = img[y - y_margin:y + h + y_margin, x - x_margin:x + w + x_margin]
            #face_gray = gray[y:y+h, x:x+w]
            #cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)
            #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Crop the detected face with a small margin

            # Resize the cropped face to a standard size
            target_size = (250, 250)
            face_resized = cv2.resize(face_gray, target_size)
            
           
            # Expand dimensions to make a batch of size 1 (required by the model)
            face_batch = np.expand_dims(face_resized, axis=0) #commented out
            #face_batch = np.expand_dims(face_batch, axis=3)  # Add channel dimension
            
            print("face shape",face_resized.shape)
            #print("face batch",face_batch)

            # Initialize LabelEncoder
            le = joblib.load('label_encoder.pkl')
            # Make prediction using the loaded model
            predictions = loaded_model.predict(face_batch)
            
            print("predictions",predictions)
            # Convert prediction to label (or name)
            predicted_label_encoded = np.argmax(predictions)  # Assuming output is one-hot encoded
            print("encoded",predicted_label_encoded)
            predicted_label = le.inverse_transform([predicted_label_encoded])[0]

            print("label value predicted", predicted_label)

            image_filename = f'{predicted_label}.jpg'
            os.makedirs("new_images", exist_ok=True)
            # Construct the full file path
            filepath = os.path.join("new_images", image_filename)
            # Save the image
            cv2.imwrite(filepath, face_resized)
            
            # Get the corresponding name for the predicted label
            name = find_names(str(predicted_label))
           # cv2.putText(gray, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Get the current date and time # code for saving attendence.csv
            """current_datetime = datetime.datetime.now()

            # Print the current date and time
            print("Current date and time:", current_datetime)

            date = current_datetime.strftime("%Y-%m-%d")
            time = current_datetime.strftime("%H:%M:%S")
            # Write data to CSV file

            with open('attendence.csv', 'a', newline='') as csvfile:
                fieldnames = ['id', 'name', 'date', 'time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header if the file is empty
                if csvfile.tell() == 0:
                    writer.writeheader()

                # Write data
                writer.writerow({'id': predicted_label, 'name': name, 'date': date, 'time': time})"""
                  #save_image(image_data, form_data)

            #Calling save_form_data to save the form details
            #save_form_data(form_data)
                # Construct response JSON object with message and name
            response_data = {
                'message': 'Image saved successfully!',
                'name': name,  # Include the name in the response
                'id': predicted_label
            }

            # Return the response as JSON
            return jsonify(response_data), 200


        else:
            print("No face detected")
            return "Invalid request", 400
        #Calling Save_image function , send image data and form data to fetch name and id to label the image data
       
       # return 'Image saved successfully!', 200
    else:
        print("response is 400")
        return "Invalid request", 400


@app.route('/save_attendence_in_csv', methods=['POST'])
def save_attendence_in_csv():
    print("saving attendence in csv")
    json_data = request.json  # Accessing jsonData from the request JSON payload
    if json_data:
        print("json data not none")
        name = json_data.get("name")  # Accessing image_data from jsonData
        id = json_data.get("id")   # Remove 'data:image/jpeg;base64,' prefix

        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Print the current date and time
        print("Current date and time:", current_datetime)

        date = current_datetime.strftime("%Y-%m-%d")
        time = current_datetime.strftime("%H:%M:%S")

        # Check if the attendance entry already exists
        if is_attendence_saved(id, name, date):
            print("Attendence for", name, "on", date, "is already saved.")

            message = f"Attendance for {name} on {date} is already saved."
    
            response_data = {
                'message': message
            }

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


def is_attendence_saved(id, name, date):
    try:
        with open('csv_files/attendence.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == id and row[1] == name and row[2] == date:
                    return True
                
    except FileNotFoundError:
        return False  # Return False if the file doesn't exist
    return False

#Saving the form data 
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
    
    print("saved")

    return 'Form data saved successfully!', 200

if __name__ == '__main__':
    app.run(debug=True)