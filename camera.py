import cv2

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
cv2.destroyAllWindows()
