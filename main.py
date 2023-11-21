import cv2 as cv
import numpy as np
import tensorflow as tf

model_path = "/home/emre/Projekte/objectdetection/data/test/keras/keras_model.h5"
model = tf.keras.models.load_model(model_path)

lable_path = "/home/emre/Projekte/objectdetection/data/test/keras/labels.txt"
with open(lable_path, "r") as file:
    labels = file.read().strip().split('\n')

confidence_threshold = 0.6 

def detection(frame, objects):
    offset = 10  
    roi = frame
    if len(objects) != 0:
        
        x,y,w,h = objects
        x -= offset
        y -= offset
        w += 2 * offset
        h += 2 * offset
        roi = frame[y:y+h, x:x+w]
    cv.imshow('ROI', roi)

    
    if roi is not None and roi.any():
        input_image = cv.resize(roi, (224, 224))
        input_image = input_image / 255.0  # Normalize the input image

        # Perform object detection
        predictions = model.predict(np.expand_dims(input_image, axis=0))[0]
        for i, confidence in enumerate(predictions):
            if confidence > confidence_threshold:
                class_label = labels[i]
                text = f"{class_label}: {confidence:.2f}"
                cv.putText(frame, text, (10, 30 * i + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow('Detected Objects', frame)

def getContours(frame):
    objects = []

#    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur_frame = cv.GaussianBlur(gray_frame, (3,3), 0)
    edges = cv.Canny(blur_frame,100, 200) 

#    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(image=edges, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_TC89_L1)
    cv.drawContours(frame, contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    cv.imshow('frame', frame)

    for contour in contours:
        min_contour_area = 40 # Passen Sie diesen Wert an
        if cv.contourArea(contour) > min_contour_area:
            x = y = w = h = None
            x, y, w, h = cv.boundingRect(contour)
            if all(coord is not None for coord in (x, y, w, h)):
                objects.append((x, y, w, h))
    
    return objects

previous_frame = None

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("No frame available")
        break
    
    current_frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    
    cv.imshow('frame', frame)
    objects = getContours(frame)
    if objects is not None:
        detection(frame, objects[0])

    previous_frame = current_frame_gray


    key = cv.waitKey(5)
    
    if key == ord("q"):
        break
    elif key == ord("s"):
        break        

cap.release()
cv.destroyAllWindows()
