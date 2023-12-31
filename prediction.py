import tensorflow as tf
import cv2

IMG_SIZE = 145

# def prepare(filepath):
#     img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
#     # image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
#     img_array = img_array / 255
#     resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#     return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

def prepare(filepath, MouthPath='./KaggleDataSet/haarcascade_mcs_mouth.xml',
                  face_cas_path = './KaggleDataSet/haarcascade_frontalface_default.xml',
                  eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')):
    result_m_e = [0, 0]
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    face_cascade = cv2.CascadeClassifier(face_cas_path)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_color = img[y:y + h, x:x + w]
        roi_gray = gray[y:y + h, x:x + w]

        mouth_cascade = cv2.CascadeClassifier(MouthPath)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 11)

        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)

            mouth_img = roi_color[my:my + mh, mx:mx + mw]
            resized_array = cv2.resize(mouth_img, (IMG_SIZE, IMG_SIZE))
            result_m_e[0] = resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            cv2.imshow('mouth detected image', mouth_img)
            cv2.waitKey(10)
            print("mouth detection is successful")
            break
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            eye_img = roi_color[ey:ey + eh, ex:ex + ew]
            resized_array = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
            result_m_e[1] = resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            cv2.imshow('eyes detected image', eye_img)
            cv2.waitKey(100)
            print("eyes detection is successful")
            break

        return result_m_e


model_e = tf.keras.models.load_model("./open_closed.model")
model_m = tf.keras.models.load_model("./yawn_no_yawn.model")


a = prepare("./KaggleDataSet/temp/no_yawn/me.jpg")
prediction1 = model_m.predict(a[0])[0][0]
prediction1 = round(prediction1)
print("mouth result(no_yawn) -> 1")
print(prediction1)
prediction2 = model_e.predict(a[1])[0][0]
prediction2 = round(prediction2)
print("eye result(open) -> 1")
print(prediction2)

print("122.jpg")
a = prepare("./KaggleDataSet/train/yawn/122.jpg")
prediction1 = model_m.predict(a[0])[0][0]
prediction1 = round(prediction1)
print("mouth result(yawn) -> 0")
print(prediction1)
prediction2 = model_e.predict(a[1])[0][0]
prediction2 = round(prediction2)
print("eye result(open) -> 1")
print(prediction2)


def test(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    result = resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    return result

b = test("./KaggleDataSet/train/Open/_0.jpg")
prediction3 = model_e.predict(b)[0][0]
prediction3 = round(prediction3)
print("eye result(open) -> 1")
print(prediction3)

c = test("./KaggleDataSet/train/Closed/_0.jpg")
prediction4 = model_e.predict(c)[0][0]
prediction4 = round(prediction4)
print("eye result(closed) -> 0")
print(prediction4)