import cv2
import dlib
import numpy as np
import os
import sys
import math
import time

MIN_EYE_OPEN_DIST = 5
MIN_MOUTH_OPEN_DIST = 10

class ImageLabelPair:
	def __init__(self, left_eye_label, left_eye_img, right_eye_label, right_eye_img, mouth_label, mouth_img):
		self.l_eye_label = left_eye_label
		self.l_eye_img = left_eye_img
		self.r_eye_label = right_eye_label
		self.r_eye_img = right_eye_img
		self.mouth_label = mouth_label
		self.mouth_img = mouth_img
		
def crop_eyes(img, l_eye_points, r_eye_points):
    return 

def crop_mouth(img, mouth_points):
    height = mouth_points[3].y - mouth_points[10].y
    width = mouth_points[0].x - mouth_points[7].x
    start_x = mouth_points[0].x
    start_y = mouth_points[3].y

    return img[start_x:start_x - width, start_y:start_y - height]

def get_eyes_mouth(file):
    return left_eyes, right_eyes, mouths

def euclidean_distance(p0, p1): # both tuple coordinate
    return math.sqrt((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2)

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
        left_eye_points = [point for point in landmarks.parts()[36:42]]
        right_eye_points = [point for point in landmarks.parts()[42:48]]
        outer_mouth_points = [point for point in landmarks.parts()[48:61]]
        inner_mouth_points = [point for point in landmarks.parts()[61:68]]
        
        # 눈, 입 상태 확인
        print("========================")
        print("File: " + img_file)
        
        #left_eye_opened = is_eye_opened(left_eye_points)
        #print("is left eye opened?: " + str(left_eye_opened))
        #
        #right_eye_opened = is_eye_opened(right_eye_points)
        #print("is right eye opened?: " + str(right_eye_opened))
        
        mouth_opened = is_mouth_opened(inner_mouth_points)
        cropped_mouth = crop_mouth(gray_img, outer_mouth_points)
        print("is mouth opened?: " + str(mouth_opened))
        cv2.imshow("mouth", cropped_mouth)
        time.sleep(1)

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
