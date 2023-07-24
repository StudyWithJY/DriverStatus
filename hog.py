import cv2
import dlib
import numpy as np
import os
import sys

def is_eye_closed(eye_points):
    # TODO

def is_mouth_open(mouth_points):
    # TODO

directory = sys.argv[1]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor()

for img_file in os.listdir(directory):
    img = cv2.imread(os.path.join(directory, img_file), cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 인식 진행
    faces = detector(gray_img)
    
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        landmarks = predictor(gray_img, face)
    
        # 눈 영역 추출
        left_eye = np.array([(point.x, point.y) for point in landmarks.parts()[36:42]])
        right_eye = np.array([(point.x, point.y) for point in landmarks.parts()[42:48]])
        
        # 입 영역 추출
        mouth = np.array([(point.x, point.y) for point in landmarks.parts()[60:68]])
        
        # 눈과 입 상태를 확인한다.
        left_eye_closed = is_eye_closed(left_eye)
        right_eye_closed = is_eye_closed(right_eye)
        mouth_open = is_mouth_open(mouth)
    
        print(f"왼쪽 눈 감김: {left_eye_closed}, 오른쪽 눈 감김: {right_eye_closed}, 입 벌림: {mouth_open}")
       
    # 결과 이미지 출력
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
