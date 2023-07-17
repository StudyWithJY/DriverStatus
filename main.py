# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
import sys

labels = os.listdir("./KaggleDataSet/train")
# print(labels)
plt.imshow(plt.imread("./KaggleDataSet/train/Closed/_0.jpg"))
# plt.show()
plt.imshow(plt.imread("./KaggleDataSet/train/yawn/1.jpg"))
# plt.show()



def face_for_yawn(direc="./KaggleDataSet/train",
                  MouthPath='./KaggleDataSet/haarcascade_mcs_mouth.xml',
                  face_cas_path = './KaggleDataSet/haarcascade_frontalface_default.xml',
                  eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
):
    yaw_noyawn = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi_color = img[y:y + h, x:x + w]
                roi_gray = gray[y:y + h, x:x + w]

                mouth_cascade = cv2.CascadeClassifier(MouthPath)
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
                mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 11)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                for (mx, my, mw, mh) in mouth:
                    cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)
                    mouth_img = roi_color[my:my + mh, mx:mx + mw]
                    resized_array = cv2.resize(mouth_img, (IMG_SIZE, IMG_SIZE))
                    yaw_noyawn.append([resized_array, class_num1])

                cv2.imshow('face, eyes and mouth detected image', mouth_img)
                cv2.waitKey(10)
                print("Face detection is successful")



    return yaw_noyawn

# def face_for_eye(direc="./KaggleDataSet/train",
#                   MouthPath='./KaggleDataSet/haarcascade_mcs_mouth.xml',
#                   face_cas_path = './KaggleDataSet/haarcascade_frontalface_default.xml',
#                   eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# ):
#     open_closed = []
#     IMG_SIZE = 145
#     #ToDo: category에 open, closed 얼굴 사진 넣기
#     categories = ["yawn", "no_yawn"]
#     for category in categories:
#         path_link = os.path.join(direc, category)
#         class_num1 = categories.index(category) + 2
#         print(class_num1)
#         for image in os.listdir(path_link):
#             image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
#             face_cascade = cv2.CascadeClassifier(face_cas_path)
#             gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 img = cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 roi_color = img[y:y + h, x:x + w]
#                 roi_gray = gray[y:y + h, x:x + w]
#
#                 mouth_cascade = cv2.CascadeClassifier(MouthPath)
#                 eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
#                 mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 11)
#
#                 for (ex, ey, ew, eh) in eyes:
#                     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#                     eye_img = roi_color[ey:ey + eh, ex:ex + ew]
#                     resized_array = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
#                     open_closed.append([resized_array, class_num1])
#                     cv2.imshow('face, eyes and mouth detected image', eye_img)
#                     cv2.waitKey(100)
#                     print("Face detection is successful")
#
#                 for (mx, my, mw, mh) in mouth:
#                     cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)
#
#
#     return open_closed

# yawn_no_yawn = face_for_yawn()


def get_eye(dir_path="./KaggleDataSet/train"):
    labels = ['Closed', 'Open']
    IMG_SIZE = 145
    open_closed = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        class_num +=2
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                open_closed.append([resized_array, class_num])
                cv2.imshow('face, eyes and mouth detected image', img_array)
                cv2.waitKey(100)
                print("Face detection is successful")
            except Exception as e:
                print(e)
    return open_closed

open_closed = get_eye()