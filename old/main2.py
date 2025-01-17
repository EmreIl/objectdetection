import cv2 as cv 
import numpy as np
import sys
from matplotlib import pyplot as plt
from keras.models import load_model
import os
import tensorflow as tf

model_path = "/home/emre/Projekte/objectdetection/data/test/keras/keras_model.h5"

# For detection2 method 
#model = tf.saved_model.load(model_path)

# For detection method
model = load_model(model_path)


lable_path = "/home/emre/Projekte/objectdetection/data/test/keras/labels.txt"
with open(lable_path, 'r') as f:
    labels = f.read().strip().split('\n')


capture = cv.VideoCapture(0)

if not capture.isOpened():
    print(f"cannont open camera", file = sys.stderr)
    sys.exit(10)


def detection2(image):
    input_tensor = cv.GaussianBlur(image, (3,3), 0)
    input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    input_tensor = tf.image.resize(input_tensor, (224, 224))
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add a batch dimension
    input_tensor = tf.cast(input_tensor, tf.float32)
    return model(input_tensor)

def detection(image, model, labels):
    image = cv.resize(image, (224,224), interpolation=cv.INTER_AREA)
    frame_array = np.expand_dims(image, axis=0)
#    frame_array = np.asarray(frame_array, dtype=np.float32).reshape(1, 224, 224, 3)
    frame_array = (frame_array / 127.5) -1
    
    prediction = model.predict(frame_array)

    confidence_threshold = 0.5  # Adjust this threshold as needed
    filtered_predictions = [p for p in prediction[0] if p >= confidence_threshold]

    class_id = np.argmax(filtered_predictions[0])
    text = labels[class_id]
    return text
    

def getContours(frame):
    objects = []

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    gray_frame = cv.cvtColor(hsv, cv.COLOR_BGR2GRAY)
    blur_frame = cv.GaussianBlur(gray_frame, (3,3), 0)
    edges = cv.Canny(blur_frame,100, 200) 

#    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(image=edges, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    for contour in contours:
        min_contour_area = 40 # Passen Sie diesen Wert an
        if cv.contourArea(contour) > min_contour_area:
            x, y, w, h = cv.boundingRect(contour)
            objects.append((x, y, w, h))
    
    return objects


while True:
    frameAvailable, frame = capture.read()

    if not frameAvailable:
        print(f"no frame availabe", file = sys.stderr)
        break

    '''obj = getContours(frame)

    for ob in obj:
        x,y,w,h = ob

    roi = frame[y:y+h, x:x+w]'''

    text = detection(frame, model, labels)
 #   cv.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('Object Detection', frame)
    key = cv.waitKey(5)
    
    if key == ord("q"):
        break
    elif key == ord("s"):
        break

capture.release()
cv.waitKey(0)
cv.destroyAllWindows()
