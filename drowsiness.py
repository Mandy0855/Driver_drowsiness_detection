import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer
import imutils
import os

# Initialize Pygame mixer
mixer.init()

# Load a sound file ("music.wav")
sound_path = "music.wav"
if os.path.exists(sound_path):
    mixer.music.load(sound_path)
else:
    print(f"Sound file '{sound_path}' not found. Please check the path.")
    exit()

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect drowsiness
def detect_drowsiness(gray, detector, predictor, frame):
    global flag
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Draw contours around the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Drowsiness check based on EAR threshold
        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Play sound alert only if not already playing
                if not mixer.music.get_busy():
                    mixer.music.play()
        else:
            flag = 0
            # Stop alarm if user is alert
            mixer.music.stop()

# Threshold and frame check values
thresh = 0.25
frame_check = 20

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()

# Specify the path to the shape predictor model file
predictor_path = "C:/Users/polep/Desktop/Driver Drowsiness detection/models-20230921T010446Z-001/models/shape_predictor_68_face_landmarks.dat"
if os.path.exists(predictor_path):
    predictor = dlib.shape_predictor(predictor_path)
else:
    print(f"Model file '{predictor_path}' not found. Please check the path.")
    exit()

# Define left and right eye landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize a flag for drowsiness detection
flag = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect drowsiness
    detect_drowsiness(gray, detector, predictor, frame)
    
    # Display the processed frame
    cv2.imshow("Frame", frame)
    
    # Check for the 'q' key to quit the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the webcam and close OpenCV windows
cv2.destroyAllWindows()
cap.release()
mixer.music.stop()
