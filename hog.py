import cv2
import dlib
import numpy as np

def is_eye_closed(eye_points):
    # 눈 가로 비율
    eye_width = np.linalg.norm(eye_points[0] - eye_points[3])
    
    # 눈 세로 비율
    eye_height = (np.linalg.norm(eye_points[1] - eye_points[5]) + np.linalg.norm(eye_points[2] - eye_points[4])) / 2
    
    # 눈 비율을 계산한다.
    ratio = eye_height / eye_width

    # 비율에 따라 눈을 감았는지 확인한다.
    if ratio > 0.25:
        return False
    else:
        return True

def is_mouth_open(mouth_points):
    # 입 너비 비율
    mouth_width = np.linalg.norm(mouth_points[0] - mouth_points[3])

    # 입 세로 비율
    mouth_height = np.linalg.norm(mouth_points[1] - mouth_points[5])

    # 입 비율을 계산한다.
    ratio = mouth_height / mouth_width

    # 입 벌어짐을 확인한다.
    if ratio > 0.3:
        return True
    else:
        return False

# 이미지 파일 경로를 지정합니다.
IMAGE_PATH = "데이터 필요!"  # 이미지 파일 경로를 작성해주세요.

# dlib의 얼굴 인식 및 랜드마크 모델 파일을 불러옵니다.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("데이터 필요!")  # 랜드마크 모델 파일 경로를 작성해주세요.

# 이미지를 읽고 그레이스케일로 변환합니다.
img = cv2.imread(IMAGE_PATH)
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
   
# 결과 이미지 출력ß
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
