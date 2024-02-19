import cv2
import numpy as np
import tensorflow.keras.models

# Video Settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

model = tensorflow.keras.models.load_model("model/CNN_model.h5")


def prediction(image):
    image = cv2.resize(image, (32, 32))
    image = image / 255
    image = image.reshape(1, 32, 32, 1)
    predict = model.predict(image)
    class_predict = np.argmax(predict, axis=1)
    result = class_predict[0]
    print(predict)
    prob = np.amax(predict)
    if prob < 0.75:
        result = 0
        prob = 0
    return result, prob


while True:
    ret, frame = cap.read()
    frame_copy = frame.copy()

    bbox_size = (60, 60)
    bbox = [(int(width // 2 - bbox_size[0] // 2), int(height // 2 - bbox_size[1] // 2)),
            (int(width // 2 + bbox_size[0] // 2), int(height // 2 + bbox_size[1]))]

    image_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (200, 200))
    cv2.imshow("Image", img_gray)

    result, prob = prediction(img_gray)
    cv2.putText(frame_copy, str(result), (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_copy, str(prob), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 3)

    cv2.imshow("frame", frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
