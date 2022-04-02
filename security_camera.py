import cv2
import time
import datetime

# access webcam
cap = cv2.VideoCapture(0)

# setup cascade classifier
# Passing in directory of where the classifiers exist
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")

# logic for only recording/saving a video when a person is in the camera
detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

# Logic for video RECORDING
# cap.get(3) gives you capture width as a floating point value - using int to round it
# cap.get(4) gives you capture height as a floating point value - using int to round it
frame_size = (int(cap.get(3)), int(cap.get(4)))

# setup 4-character code - unique identifier for the specific format our video is saved as (.mp4 format)
# passes mp4v string as 4 parameters for the function
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


while True:
    # reading one frame from webcam then displays on the screen
    # _ is a placeholder for a variable that we don't care about
    _, frame = cap.read()

    # convert captured image to grayscale
    # grayscale is necessary for the classification
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # create classifier for grayscale image
    # Returns a list of positions (x, y, width, and height) of all faces within our capture frame
    # 1.3 is the scame factor - determines the accuracy and speed of algorithm (should be between 1.0 - 1.5+)
    # Note: the closer to 1.0, the more accurate, but slower it will run
    # 5 is minimum number of neighbors - detects x-amount of faces within the capture and "boxes around them"
    # Setting 5 is the number of boxes that need to surround a figure (face) for the program to recognize it's actually a face
    # Minimum neighbors can be between 3 - 6, where higher number is less faces detected, lower number is more faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    # Check if a body or face is within the frame
    # done by looking at length of faces and length of bodies (ebcause they're lists of locations)
    # if we have at least 1 body or face, we start recording
    if len(faces) + len(bodies) > 0:
        # Logic: IF we detect a face/body, BUT we've already detected a face or body, then DO NOT stop recording
        # Basically, if we detect a new face/body while already recording, just keep recording current video
        if detection:
            timer_started = False
        else:
            # If we haven't already detected something, then we want to start a new recording
            detection = True
            # output stream for where we want to write all our content to
            # close output stream once we want to save the video
            # Passing video name, 4 character code, 20 = frame rate, and frame size
            # We set the videos name as the current date/time
            # datetime formatting: day-month-year-hour-minute-second
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(
                f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started Recording!")
    elif detection:
        # If we were just recording something, we first check if the timer was started
        if timer_started:
            # If the timer has been started, we check if the current time is 5 seconds past the last detection time
            # if it has not, the video continues
            # if it has been 5 seconds, we set detection and start timer to false and STOP the recording and everything resets
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Stopped Recording!")
        else:
            # If we did not detect a body or face, but we have detected something previously, we start timer
            # for when we should end the video stream (recording)
            timer_started = True
            # sets variable as the current time
            detection_stopped_time = time.time()

    # Ensure we are only writing to the output stream if we are detecting something
    if detection:
        out.write(frame)

    """
    #Note: Commented out because we don't care about the position of the faces/bodies, but this is good for testing/visualization
    # drawing faces on image
    # draws a (blue) rectangle over our COLOR image ; note: color image is in BGR
    # (x, y) is top-left corner of rectangle; (x+width, y+height) is bottom-right corner of rectangle
    # 3 is rectangle line thickness (3 pixels thick)
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)
     for (x, y, width, height) in bodies:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 3)
    """

    # opens window to show video capture. Frame is titled "Camera"
    cv2.imshow('Camera', frame)

    # Pressing q-key will end and close the program
    if cv2.waitKey(1) == ord('q'):
        break

# when the program is closed, it will save our recording
out.release()
# release camera resources
cap.release()
cv2.destroyAllWindows()
