import cv2 as cv 
import numpy as np
import sys
from matplotlib import pyplot as plt
from keras.models import load_model
import os
import tensorflow as tf

model_path = "/home/emre/Projekte/objectdetection/data/model.savedmodel"

model = tf.saved_model.load(model_path)

lable_path = "/home/emre/Projekte/objectdetection/data/labels.txt"
with open(lable_path, 'r') as f:
    labels = f.read().strip().split('\n')
    print(labels)

capture = cv.VideoCapture(0)

if not capture.isOpened():
    print(f"cannont open camera", file = sys.stderr)
    sys.exit(10)

while True:
    frameAvailable, frame = capture.read()

    if not frameAvailable:
        print(f"no frame availabe", file = sys.stderr)
        break

    cv.imshow('Object Detection', frame)

    key = cv.waitKey(5)
    
    if key == ord("q"):
        break
    elif key == ord("s"):
        break

capture.release()
cv.waitKey(0)
cv.destroyAllWindows()
