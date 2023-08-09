import os
import pickle

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.models import Sequential


def face_for_yawn(direc="./KaggleDataSet/train",
                  MouthPath='./KaggleDataSet/haarcascade_mcs_mouth.xml',
                  face_cas_path='./KaggleDataSet/haarcascade_frontalface_default.xml',
                  eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                  ):
    yawn_noyawn = []
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
                # img = cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi_color = image_array[y:y + h, x:x + w]
                roi_gray = gray[y:y + h, x:x + w]

                mouth_cascade = cv2.CascadeClassifier(MouthPath)
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
                mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 11)

                # for (ex, ey, ew, eh) in eyes:
                # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                for (mx, my, mw, mh) in mouth:
                    # cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)
                    mouth_img = roi_color[my:my + mh, mx:mx + mw]
                    resized_array = cv2.resize(mouth_img, (IMG_SIZE, IMG_SIZE))
                    yawn_noyawn.append([resized_array, class_num1])
                    # cv2.imshow('mouth detected image', mouth_img)
                    # cv2.waitKey(10)
                    # print("mouth detection is successful")

    return yawn_noyawn


def face_for_eye(direc="./KaggleDataSet/train",
                 MouthPath='./KaggleDataSet/haarcascade_mcs_mouth.xml',
                 face_cas_path='./KaggleDataSet/haarcascade_frontalface_default.xml',
                 eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                 ):
    open_closed = []
    IMG_SIZE = 145

    categories = ["Open", "Closed"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category) + 2
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # img = cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi_color = image_array[y:y + h, x:x + w]
                roi_gray = gray[y:y + h, x:x + w]

                mouth_cascade = cv2.CascadeClassifier(MouthPath)
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
                mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 11)

                for (ex, ey, ew, eh) in eyes:
                    # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    eye_img = roi_color[ey:ey + eh, ex:ex + ew]
                    resized_array = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
                    open_closed.append([resized_array, class_num1])
                    # cv2.imshow('eyes detected image', eye_img)
                    # cv2.waitKey(100)
                    # print("eyes detection is successful")

                # for (mx, my, mw, mh) in mouth:
                #     cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)

    return open_closed


# def get_eye(dir_path="./KaggleDataSet/train"):
#     labels = ['Closed', 'Open']
#     IMG_SIZE = 145
#     open_closed = []
#     for label in labels:
#         path = os.path.join(dir_path, label)
#         class_num = labels.index(label)
#         class_num += 2
#         print(class_num)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
#                 resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#                 open_closed.append([resized_array, class_num])
#                 # cv2.imshow('eyes detected image', img_array)
#                 # cv2.waitKey(10)
#                 # print("eyes detection is successful")
#             except Exception as e:
#                 print(e)
#     return open_closed


print("yawn_no_yawn 모델********************************************************************************")

is_pickle = True
if not is_pickle:
    yawn_no_yawn = face_for_yawn()  # yawn 0, no_yawn 1
    yawn_no_yawn = np.array(yawn_no_yawn)
    with open("(OpenCV)yawn_no_yawn.pickle", "wb") as f:
        pickle.dump(yawn_no_yawn, f)
else:
    with open("(OpenCV)yawn_no_yawn.pickle", "rb") as f:
        yawn_no_yawn = pickle.load(f)

print("len(yawn_no_yawn) = ", len(yawn_no_yawn))

X = []  # 데이터
y = []  # 라벨
for feature, label in yawn_no_yawn:
    X.append(feature)
    y.append(label)  # label은 class num임

X = np.array(X)
X = X.reshape(-1, 145, 145, 3)

label_bin = LabelEncoder()  # yawn 0, no_yawn 1
label_bin.fit(y)
y = label_bin.transform(y)
y = np.array(y)

seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

print("len(X_train) = ", len(X_train))
print("len(X_test) = ", len(X_test))
print("len(y_train) = ", len(y_train))
print("len(y_test) = ", len(y_test))

train_generator_m = ImageDataGenerator(rescale=1 / 255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
# test_generator_m = ImageDataGenerator(rescale=1 / 255)
train_generator_m = train_generator_m.flow(np.array(X_train), y_train, shuffle=False)
# test_generator_m = test_generator_m.flow(np.array(X_test), y_test, shuffle=False)

model_train = False

if model_train:
    model_m = Sequential()

    model_m.add(Conv2D(256, (3, 3), activation="relu", input_shape=X_train.shape[1:]))
    model_m.add(MaxPooling2D(2, 2))

    model_m.add(Conv2D(128, (3, 3), activation="relu"))
    model_m.add(MaxPooling2D(2, 2))

    model_m.add(Conv2D(64, (3, 3), activation="relu"))
    model_m.add(MaxPooling2D(2, 2))

    model_m.add(Conv2D(32, (3, 3), activation="relu"))
    model_m.add(MaxPooling2D(2, 2))

    model_m.add(Flatten())
    model_m.add(Dropout(0.5))

    model_m.add(Dense(64, activation="relu"))
    model_m.add(Dense(1, activation="sigmoid"))

    model_m.compile(loss='mean_squared_error', metrics=["accuracy"], optimizer="adam")

    model_m.summary()

    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=100)
    # print(len(test_generator_m), len(X_test), len(y_test))
    history_m = model_m.fit(train_generator_m, epochs=100, validation_data=(X_test, y_test), shuffle=True,
                            batch_size=16, steps_per_epoch=len(X_train)//16)  # steps_per_epoch는 데이터셋 개수, steps_per_epoch=len(X_train)

    accuracy = history_m.history['accuracy']
    val_accuracy = history_m.history['val_accuracy']
    loss = history_m.history['loss']
    val_loss = history_m.history['val_loss']
    epochs = range(len(accuracy))

    plt.clf()

    print("yawn_no_yawn model accuracy")
    plt.plot(epochs, accuracy, "b", label="trainning accuracy")
    plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
    plt.title("OpenCV accuracy(yawn, no_yawn)")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    print("yawn_no_yawn model loss")
    plt.plot(epochs, loss, "b", label="trainning loss")
    plt.plot(epochs, val_loss, "r", label="validation loss")
    plt.title("OpenCV loss(yawn, no_yawn)")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    model_m.save("(OpenCV)yawn_no_yawn.model")
else:
    model_m = keras.models.load_model("(OpenCV)yawn_no_yawn.model")

y_prob = model_m.predict(X_test, verbose=0)
prediction = np.round(y_prob[:, 0]).astype(int)
# print("yawn_no_yawn model prediction", prediction)

labels_new = ["yawn", "no_yawn"]

model_m.evaluate(X_test, y_test)

print(classification_report(y_test, prediction, target_names=labels_new))

print("open_closed 모델******************************************************************************************************")

is_pickle = True
if not is_pickle:
    open_closed = face_for_eye()  # closed 2, open 3
    open_closed = np.array(open_closed)
    with open("(OpenCV)open_closed.pickle", "wb") as f:
        pickle.dump(open_closed, f)
else:
    with open("(OpenCV)open_closed.pickle", "rb") as f:
        open_closed = pickle.load(f)

print("len(open_closed) = ", len(open_closed))

X = []  # 데이터
y = []  # 라벨
for feature, label in open_closed:
    X.append(feature)
    y.append(label)

X = np.array(X)
X = X.reshape(-1, 145, 145, 3)

label_bin = LabelEncoder()  # closed 0, open 1
label_bin.fit(y)
y = label_bin.transform(y)
y = np.array(y)

seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

print("len(X_train) = ", len(X_train))
print("len(X_test) = ", len(X_test))
print("len(y_train) = ", len(y_train))
print("len(y_test) = ", len(y_test))

train_generator_e = ImageDataGenerator(rescale=1 / 255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
# test_generator_e = ImageDataGenerator(rescale=1 / 255)
train_generator_e = train_generator_e.flow(np.array(X_train), y_train, shuffle=False)
# test_generator_e = test_generator_e.flow(np.array(X_test), y_test, shuffle=False)

model_train = False

if model_train:
    model_e = Sequential()

    model_e.add(Conv2D(256, (3, 3), activation="relu", input_shape=X_train.shape[1:]))
    model_e.add(MaxPooling2D(2, 2))

    model_e.add(Conv2D(128, (3, 3), activation="relu"))
    model_e.add(MaxPooling2D(2, 2))

    model_e.add(Conv2D(64, (3, 3), activation="relu"))
    model_e.add(MaxPooling2D(2, 2))

    model_e.add(Conv2D(32, (3, 3), activation="relu"))
    model_e.add(MaxPooling2D(2, 2))

    model_e.add(Flatten())
    model_e.add(Dropout(0.5))

    model_e.add(Dense(64, activation="relu"))
    model_e.add(Dense(1, activation="sigmoid"))

    model_e.compile(loss='mean_squared_error', metrics=["accuracy"], optimizer="adam")

    model_e.summary()

    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=100)
    history_e = model_e.fit(train_generator_e, epochs=100, validation_data=(X_test, y_test), shuffle=True,
                            batch_size=16, steps_per_epoch=len(X_train)//16)  # steps_per_epoch = 725

    accuracy = history_e.history['accuracy']
    val_accuracy = history_e.history['val_accuracy']
    loss = history_e.history['loss']
    val_loss = history_e.history['val_loss']
    epochs = range(len(accuracy))

    plt.clf()

    print("open_closed model accuracy")
    plt.plot(epochs, accuracy, "b", label="trainning accuracy")
    plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
    plt.title("OpenCV accuracy(open, closed)")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    print("open_closed model loss")
    plt.plot(epochs, loss, "b", label="trainning loss")
    plt.plot(epochs, val_loss, "r", label="validation loss")
    plt.title("OpenCV loss(open, closed)")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    model_e.save("(OpenCV)open_closed.model")
else:
    model_e = keras.models.load_model("(OpenCV)open_closed.model")

y_prob = model_e.predict(X_test, verbose=0)
prediction = np.round(y_prob[:, 0]).astype(int)
# print("open_closed model prediction", prediction)

labels_new = ["Closed", "Open"]

model_e.evaluate(X_test, y_test)

print(classification_report(y_test, prediction, target_names=labels_new))
