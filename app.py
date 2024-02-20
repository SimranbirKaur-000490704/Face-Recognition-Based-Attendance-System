import base64
import csv
import time
from flask import Flask, render_template, Response, request
import os
import uuid
import cv2
import numpy as np
#import camera
#print(cv2.__version__)

app = Flask("FRAMS")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


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


# Load your webcam capture script here
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
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


# Load your webcam capture script here
"""def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detect faces in the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h),  (0, 110, 0), 2)
                
                face = frame[y:y+h, x:x+w]

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', face)
            face = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + face + b'\r\n')"""


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

        #Calling save_form_data to save the form details
        save_form_data(form_data)

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

    #Calling generate unique id function
    #image_id = generate_unique_id()

    #Creating label for the image
    id = form_data.get("Student Id") 
    name = form_data.get("Student Name")
    image_id =  id+"_"+name

    #Creating a directory with person's name
    directory = os.path.join('images', name)

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

    #image_path = os.path.join(directory, f'{image_id}.jpg')
    #cv2.imwrite(image_path, cropped_image)

    # Save the cropped image to a folder with the unique ID as the filename
   # image_path = os.path.join('images', f'{image_id}.jpg')
    #cv2.imwrite(image_path, cropped_image)


#Function to detect face and crop the image, resize it and return
def image_cleaning_resizing(image):
        # Ensure that the input image is in the correct format
    if isinstance(image, str):
        # Load the image from file
        print("aaaaa----")

        image = cv2.imread(image)
    elif not isinstance(image, np.ndarray):
        # If image is not a NumPy array, return None
        print("bbbbbbb----")

        return None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Crop and save each detected face
    for i, (x, y, w, h) in enumerate(faces):
        # Crop the detected face with a small margin
        margin = 0.2
        x_margin = int(w * margin)
        y_margin = int(h * margin)
        face = image[y - y_margin:y + h + y_margin, x - x_margin:x + w + x_margin]

        # Resize the cropped face to a standard size
        target_size = (250, 250)
        face_resized = cv2.resize(face, target_size)

        return face_resized

#Saving the form data 
def save_form_data(form_data):
    csv_file_path = 'form_data.csv'
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

        for record in existing_records:
            if record['Student Id'] != form_data['Student Id']:
                writer.writerow(record)

        # Write the updated record if it exists
        if record_exists:
            writer.writerow(form_data)

    return 'Form data saved successfully!', 200

    
# This can be deleted , Function to generate a unique ID for each image
def generate_unique_id():
    return str(uuid.uuid4())

if __name__ == '__main__':
    app.run(debug=True)

    
    