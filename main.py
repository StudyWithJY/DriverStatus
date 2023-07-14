# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
import sys

# labels = os.listdir("./KaggleDataSet/train")
# # print(labels)
# plt.imshow(plt.imread("./KaggleDataSet/train/Closed/_0.jpg"))
# # plt.show()
# plt.imshow(plt.imread("./KaggleDataSet/train/yawn/1.jpg"))
# # plt.show()


# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#
# # 이미지 불러오기
# img = cv2.imread('./KaggleDataSet/me.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 얼굴 찾기
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#     # 눈 찾기
#     roi_color = img[y:y + h, x:x + w]
#     roi_gray = gray[y:y + h, x:x + w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
#
# # 영상 출력
# cv2.imshow('image', img)
#
# key = cv2.waitKey(0)
# cv2.destroyAllWindows()

MouthPath = './KaggleDataSet/haarcascade_mcs_mouth.xml'

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(MouthPath)

image = cv2.imread('./KaggleDataSet/me.jpg')
# cv2.imshow('Original image', image)
# cv2.waitKey(5000)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(image, 1.4, 4)
for(x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    roi_gray = gray_image[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(gray_image, 1.3, 5)
    mouth = mouth_cascade.detectMultiScale(gray_image, 1.5, 11)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    for (mx, my, mw, mh) in mouth:
        cv2.rectangle(image, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)

cv2.imshow('face, eyes and mouth detected image', image)
cv2.waitKey()
print("Face, eye and mouth detection is successful")