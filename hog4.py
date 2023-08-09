import dlib
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.models import Sequential


def face_for_yawn(directory = "./KaggleDataSet/train", detector = dlib.get_frontal_face_detector()):

    yawn_noyawn = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]

    for category in categories:
        path_link = os.path.join(directory, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)

                x = landmarks.parts()[48]  #left
                x = x.x
                y = landmarks.parts()[52]  #top
                y = y.y
                w = landmarks.parts()[54].x    #right
                h = landmarks.parts()[57].y   #bottom

                # 좌측 상단: x, y
                # 우측 하단: w, h
                # print(x, y, w, h)

                mouth_img = image_array[y-2:h+2, x:w]     # [startY:endY, startX:endX]

                resized_array = cv2.resize(mouth_img, (IMG_SIZE, IMG_SIZE))
                yawn_noyawn.append([resized_array, class_num1])
                cv2.imshow('mouth detected image', mouth_img)
                cv2.waitKey(10)
                print("mouth detection is successful")

    return yawn_noyawn




def face_for_eye(directory = "./KaggleDataSet/train", detector = dlib.get_frontal_face_detector()):

    open_closed = []
    IMG_SIZE = 145
    categories = ["Open", "Closed"]

    for category in categories:
        path_link = os.path.join(directory, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)

                x = landmarks.parts()[36]  #left
                x = x.x
                y = landmarks.parts()[38]  #top
                y = y.y
                w = landmarks.parts()[39].x    #right
                h = landmarks.parts()[41].y   #bottom

                # 좌측 상단: x, y
                # 우측 하단: w, h
                # print(x, y, w, h)

                eye_img = image_array[y-2:h+2, x:w]     # [startY:endY, startX:endX]

                resized_array = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
                open_closed.append([resized_array, class_num1])
                cv2.imshow('mouth detected image', eye_img)
                cv2.waitKey(10)
                print("mouth detection is successful")

    return open_closed


# yawn_no_yawn 모델********************************************************************************
yawn_no_yawn = face_for_yawn()  # yawn 0, no_yawn 1
yawn_no_yawn = np.array(yawn_no_yawn)

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

train_generator_m = ImageDataGenerator(rescale=1 / 255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
test_generator_m = ImageDataGenerator(rescale=1 / 255)
train_generator_m = train_generator_m.flow(np.array(X_train), y_train, shuffle=False)
test_generator_m = test_generator_m.flow(np.array(X_test), y_test, shuffle=False)

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

model_m.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

model_m.summary()

history_m = model_m.fit(train_generator_m, epochs=10, validation_data=test_generator_m, shuffle=True,
                        validation_steps=len(test_generator_m),
                        steps_per_epoch=len(X_train))  # steps_per_epoch는 데이터셋 개수, steps_per_epoch = 240

accuracy = history_m.history['accuracy']
val_accuracy = history_m.history['val_accuracy']
loss = history_m.history['loss']
val_loss = history_m.history['val_loss']
epochs = range(len(accuracy))

plt.clf()

print("yawn_no_yawn model accuracy")
plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()

print("yawn_no_yawn model loss")
plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()
plt.show()

model_m.save("yawn_no_yawn.model")

prediction = model_m.predict_classes(X_test)
print("yawn_no_yawn model prediction", prediction)

labels_new = ["yawn", "no_yawn"]

print(classification_report(y_test, prediction, target_names=labels_new))

# open_closed 모델******************************************************************************************************
open_closed = face_for_eye()  # closed 2, open 3
open_closed = np.array(open_closed)

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

train_generator_e = ImageDataGenerator(rescale=1 / 255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
test_generator_e = ImageDataGenerator(rescale=1 / 255)
train_generator_e = train_generator_e.flow(np.array(X_train), y_train, shuffle=False)
test_generator_e = test_generator_e.flow(np.array(X_test), y_test, shuffle=False)

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

model_e.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

model_e.summary()

history_e = model_e.fit(train_generator_e, epochs=10, validation_data=test_generator_e, shuffle=True,
                        validation_steps=len(test_generator_e), steps_per_epoch=len(X_train))  # steps_per_epoch = 725

accuracy = history_e.history['accuracy']
val_accuracy = history_e.history['val_accuracy']
loss = history_e.history['loss']
val_loss = history_e.history['val_loss']
epochs = range(len(accuracy))

plt.clf()

print("open_closed model accuracy")
plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()

print("open_closed model loss")
plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()
plt.show()

model_e.save("open_closed.model")

prediction = model_e.predict_classes(X_test)
print("open_closed model prediction", prediction)

labels_new = ["Closed", "Open"]

print(classification_report(y_test, prediction, target_names=labels_new))

