import os
import cv2

FACE_PATH = './KaggleDataSet/haarcascade_frontalface_default.xml'
MOUTH_PATH = './KaggleDataSet/haarcascade_mcs_mouth.xml'
EYE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'

# get frame of video and detect eyes and mouthes
def get_eye_yawn(frame):
    # read frame buffer from memory and return its matrix
    frame_matrix = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE) 
    # define how to detect faces
    face_cascade = cv2.CascadeClassifier(FACE_PATH)
    # take faces from frame, face maybe multiple
    faces = face_cascade.detectMultiScale(frame_matrix, 1.3, 5)
    
    # returning list
    eye_mouth_pairs = []
    for (fx, fy, fw, fh) in faces:
        # crop face
        croped_face = cv2.clipLine(frame_matrix, (fx, fy), (fx + fw, fy + fh))

        eye_mouth_pair = []

        # Detect eyes
        eye_cascade = cv2.CascadeClassifier(EYE_PATH)
        eyes = eye_cascade.detectMultiScale(croped_face, 1.3, 5)
        
        for (ex, ey, ew, eh) in eyes:
            eye = cv2.clipLine(croped_face, (ex, ey), (ex + ew, ey + eh))
            eye_mouth_pair.append(eye)
        
        # Detect mouth 
        mouth_cascade = cv2.CascadeClassifier(MOUTH_PATH)
        mouthes = mouth_cascade.detectMultiScale(croped_face, 1.3, 5)
        
        for (mx, my, mw, mh) in mouthes:
            mouth = cv2.clipLine(croped_face, (mx, my), (mx + mw, my + mh))
            eye_mouth_pair.append(mouth)

        # push into list
        eye_mouth_pairs.append(eye_mouth_pair)

    return eye_mouth_pairs

# test code
# TODO
