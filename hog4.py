import cv2
import dlib
import numpy as np
import os
import sys
import math

FACE_PATH = './KaggleDataSet/haarcascade_frontalface_default.xml'
MOUTH_PATH = './KaggleDataSet/haarcascade_mcs_mouth.xml'
EYE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'
MIN_EYE_OPEN_DIST = 5
MIN_MOUTH_OPEN_DIST = 10

# get frame of video and detect eyes and mouthes

class ImageLabelPair:
	def __init__(self, left_eye_label, left_eye_img, right_eye_label, right_eye_img, mouth_label, mouth_img):
		self.left_eye_label = left_eye_label
		self.left_eye_img = left_eye_img
		self.right_eye_label = right_eye_label
		self.right_eye_img = right_eye_img
		self.mouth_label = mouth_label
		self.mouth_img = mouth_img
		
def get_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def crop_eyes(image, faces, eyes_cascade):
    left_eyes = []
    right_eyes = []
    
    for (x, y, w, h) in faces:
        roi_gray = image[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) < 2:
            continue

        eye_1, eye_2 = eyes[0], eyes[1]

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
        
        left_eye_roi = roi_gray[left_eye[1]:left_eye[1] + left_eye[3], left_eye[0]:left_eye[0] + left_eye[2]]
        right_eye_roi = roi_gray[right_eye[1]:right_eye[1] + right_eye[3], right_eye[0]:right_eye[0] + right_eye[2]]
        
        left_eyes.append(left_eye_roi)
        right_eyes.append(right_eye_roi)
    
    return left_eyes, right_eyes

def crop_mouth(image, faces, mouth_cascade):
    mouths = []

    for (x, y, w, h) in faces:
        roi_gray = image[y:y+h, x:x+w]
        mouths.append(mouth_cascade.detectMultiScale(roi_gray))

    return mouths

def get_eyes_mouth(file):
    face_cascade = cv2.CascadeClassifier(FACE_PATH)
    eyes_cascade = cv2.CascadeClassifier(EYE_PATH)
    mouth_cascade = cv2.CascadeClassifier(MOUTH_PATH)
	
    image = cv2.imread(file)
    
    faces = get_faces(image, face_cascade)
    
    left_eyes, right_eyes = crop_eyes(image, faces, eyes_cascade)
    mouths = crop_mouth(image, faces, mouth_cascade)
    return left_eyes, right_eyes, mouths

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

        arr = []
        imgs = get_eyes_mouth(os.path.join(directory, img_file))
        arr.append(ImageLabelPair(
            left_eye_opened, 
            imgs[0][0], 
            right_eye_opened, 
            imgs[1][0], 
            mouth_opened, 
            imgs[2][0]
        ))
       
        for obj in arr:
            cv2.imshow("left eye", obj.left_eye_img)
            cv2.imshow("right eye", obj.right_eye_img)
            #cv2.imshow("mouth", obj.mouth_img)
