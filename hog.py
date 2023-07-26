import cv2
import dlib
import numpy as np
import os
import sys
import math

FACE_PATH = './KaggleDataSet/haarcascade_frontalface_default.xml'
MOUTH_PATH = './KaggleDataSet/haarcascade_mcs_mouth.xml'
EYE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'

# get frame of video and detect eyes and mouthes
def get_eye_mouth(frame):
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

MIN_EYE_OPEN_DIST = 5
MIN_MOUTH_OPEN_DIST = 10

def euclidean_distance(p0, p1): # both tuple coordinate
    return math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)

def is_eye_opened(eye_points): # list of tuple of coordinates
    # compute distance between two points of eye by euclidean
    distance0 = euclidean_distance(eye_points[1], eye_points[5])
    distance1 = euclidean_distance(eye_points[2], eye_points[4])

    if distance0 > MIN_EYE_OPEN_DIST and distance1 > MIN_EYE_OPEN_DIST:
        return True
    else:
        return False

def is_mouth_opened(mouth_points):
    distance = euclidean_distance(mouth_points[2], mouth_points[4])

    if distance > MIN_MOUTH_OPEN_DIST:
        return True
    else:
        return False

directory = sys.argv[1]
detector = dlib.get_frontal_face_detector()

for img_file in os.listdir(directory):
    img = cv2.imread(os.path.join(directory, img_file), cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 인식 진행
    # download 'https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2'
    # and decompress it
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faces = detector(gray_img)
   
    for face in faces:
        landmarks = predictor(gray_img, face)
    
        # 눈, 입 영역 추출
        left_eye_points = [(point.x, point.y) for point in landmarks.parts()[36:42]]
        right_eye_points = [(point.x, point.y) for point in landmarks.parts()[42:48]]
        mouth_points = [(point.x, point.y) for point in landmarks.parts()[61:67]]
        
        # 눈, 입 상태 확인
        print("========================")
        print("File: " + img_file)
        
        left_eye_opened = is_eye_opened(left_eye_points)
        print("is left eye opened?: " + str(left_eye_opened))
        
        right_eye_opened = is_eye_opened(right_eye_points)
        print("is right eye opened?: " + str(right_eye_opened))
        
        mouth_opened = is_mouth_opened(mouth_points)
        print("is mouth opened?: " + str(mouth_opened))
