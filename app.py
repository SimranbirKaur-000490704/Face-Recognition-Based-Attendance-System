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

# Load your webcam capture script here


"""def generate_frames():
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detect faces in the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            if len(faces) > 0:
                # If faces are detected, capture and yield only the first detected face
                x, y, w, h = faces[0]
                
                # Decode the frame bytes to a NumPy array
                frame_array = np.frombuffer(frame, dtype=np.uint8)
                frame_array = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                # Extract the face region from the frame array
                face = frame_array[y:y+h, x:x+w]
                
                # Encode the face region to JPEG format and yield it
                ret, buffer = cv2.imencode('.jpg', face)
                face_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + face_bytes + b'\r\n')
            else:
                # If no faces are detected, yield the entire frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')"""



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
    print("saveimgae method")
    image_data = request.json['image_data']

    #Calling Save_image function
    save_image(image_data)

    #save_image(image_data)
    return 'Image saved successfully!', 200

def save_image(image_data):
    print("in save images method")
    image_data = image_data.split(',')[1]  # Remove 'data:image/jpeg;base64,' prefix
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Calling image_cleaning_resizing function to resize and crop the image to get a face
    cropped_image = image_cleaning_resizing(img)

    #Calling generate unique id function
    image_id = generate_unique_id()

    # Save the cropped image to a folder with the unique ID as the filename
    image_path = os.path.join('images', f'{image_id}.jpg')
    cv2.imwrite(image_path, cropped_image)


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

    """if len(faces) > 0:
        # Get the coordinates of the first detected face
        x, y, w, h = faces[0]
        
        # Crop the region containing the face
        cropped_image = image[y:y+h, x:x+w]
        
        return cropped_image
    else:
        # If no faces are detected, return None
        return None"""
    
"""def image_cleaning_resizing(image):
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

   # cap = cv2.VideoCapture(0)
    #while True:
        #success, frame = cap.read()
       # if not success:
        #    break
    if(image):
        # Detect faces in the frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Encode frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', image)
        image = buffer.tobytes()
            
        if len(faces) > 0:
             # If faces are detected, capture and yield only the first detected face
            x, y, w, h = faces[0]
                
            # Decode the frame bytes to a NumPy array
            frame_array = np.frombuffer(image, dtype=np.uint8)
            frame_array = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
            # Extract the face region from the frame array
            face = frame_array[y:y+h, x:x+w]
                
            # Encode the face region to JPEG format and yield it
            ret, buffer = cv2.imencode('.jpg', face)
            face_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + face_bytes + b'\r\n')
        else:
            # If no faces are detected, yield the entire frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')"""

#Saving the form data in csv
@app.route('/save_form_data', methods=['POST'])
def save_form_data():
    data = request.json  # Get form data
    with open('form_data.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if csvfile.tell() == 0:  # Write header if file is empty
            writer.writeheader()
        writer.writerow(data)  # Write form data to CSV
    return 'Form data saved successfully!', 200

    
# Function to generate a unique ID for each image
def generate_unique_id():
    return str(uuid.uuid4())

if __name__ == '__main__':
    app.run(debug=True)

    
    