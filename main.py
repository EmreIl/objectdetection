import cv2 as cv 
import numpy as np
import sys
from matplotlib import pyplot as plt
from keras.models import load_model
import os
import tensorflow as tf

model_path = "/home/emre/Projekte/objectdetection/data/test/model.savedmodel"

# For detection2 method 
#model = tf.saved_model.load(model_path)

# For detection method
model = load_model(model_path)


lable_path = "/home/emre/Projekte/objectdetection/data/test/labels.txt"
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

def detection(image):
#    image = cv.GaussianBlur(image, (3,3), 0)
    image = cv.resize(image, (224,224), interpolation=cv.INTER_AREA)
    frame_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    frame_array = (frame_array / 127.5) -1
    return model.predict(frame_array)

while True:
    frameAvailable, frame = capture.read()

    if not frameAvailable:
        print(f"no frame availabe", file = sys.stderr)
        break
    
    prediction = detection(frame)

    print(prediction)

    class_id = np.argmax(prediction)
    text = labels[class_id]

    frame = cv.resize(frame, (500,500), interpolation=cv.INTER_AREA)
 #   cv.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imshow('Object Detection', frame)
    key = cv.waitKey(5)
    
    if key == ord("q"):
        break
    elif key == ord("s"):
        break

capture.release()
cv.waitKey(0)
cv.destroyAllWindows()
