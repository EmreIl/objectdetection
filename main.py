import cv2 as cv
import numpy as np
import tensorflow as tf

#Linux
#model_path = "/home/emre/Projekte/objectdetection/data/test/keras/keras_model.h5"
#Mac
model_path ="data/keras/keras_model.h5"
#model = tf.keras.models.load_model(model_path)

#model_path ="data/newModeltest/keras/keras_model.h5"
model = tf.keras.models.load_model(model_path)


#Linux
#lable_path = "/home/emre/Projekte/objectdetection/data/test/keras/labels.txt"
#Macos
lable_path ="data/keras/labels.txt"
#lable_path ="data/newModeltest/keras/labels.txt"
with open(lable_path, "r") as file:
    labels = file.read().strip().split('\n')

confidence_threshold = 0.7

def detection(objects):
    offset = 5  
    roi = frame
    if len(objects) != 0:
        
        x,y,w,h = objects
        """ x -= offset
        y -= offset
        w += 2 * offset
        h += 2 * offset """
        roi = frame[y:y+h, x:x+w]
        
    contour = np.array([(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)], dtype=np.int32)
 
    cv.imshow('Rotated ROI', roi)

    
    if roi is not None and roi.any():
        input_image = cv.resize(roi, (224, 224))
        input_image = input_image / 255.0  # Normalize the input image

        # Perform object detection
        predictions = model.predict(np.expand_dims(input_image, axis=0))[0]
        
        #input_tensor = tf.convert_to_tensor(roi, dtype=tf.uint8)
        input_tensor = tf.image.resize(roi, (224, 224))
        input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add a batch dimension
        input_tensor = tf.cast(input_tensor, tf.float32) / 255

        for i, confidence in enumerate(predictions):
            if confidence > confidence_threshold:
                class_label = labels[i]
                text = f"{class_label}: {confidence:.2f}"
                cv.putText(frame, text, (10, 30 * i + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.imshow('frame', frame)

def getlegoStones():
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Define color ranges for each color
    blue_lower = np.array([110, 50, 50])
    blue_upper = np.array([130, 255, 255])

    orange_lower = np.array([10, 100, 100])
    orange_upper = np.array([20, 255, 255])

    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 30])

    brown_lower = np.array([0, 60, 60])
    brown_upper = np.array([30, 255, 255])

    green_lower = np.array([40, 40, 40])
    green_upper = np.array([80, 255, 255])

    # Create masks
    blue_mask = cv.inRange(hsv_image, blue_lower, blue_upper)
    orange_mask = cv.inRange(hsv_image, orange_lower, orange_upper)
    black_mask = cv.inRange(hsv_image, black_lower, black_upper)
    brown_mask = cv.inRange(hsv_image, brown_lower, brown_upper)
    green_mask = cv.inRange(hsv_image, green_lower, green_upper)

def getContours():
    objects = []

#    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur_frame = cv.GaussianBlur(gray_frame, (9,9), 0)
    edges = cv.Canny(blur_frame,100, 200) 

    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv.dilate(edges, kernel, iterations=1)
    eroded_edges = cv.erode(dilated_edges, kernel, iterations=1)


#    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(image=eroded_edges, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_TC89_L1)
    cv.drawContours(frame, contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

    for contour in contours:
        min_contour_area = 40 # Passen Sie diesen Wert an
        if cv.contourArea(contour) > min_contour_area:
            x = y = w = h = None
            x, y, w, h = cv.boundingRect(contour)
            if all(coord is not None for coord in (x, y, w, h)):
                objects.append((x, y, w, h))
    
    return objects

previous_frame = None

cap = cv.VideoCapture(1)
cap.set(3, 1280)
"""cap.set(cv.CAP_PROP_AUTOFOCUS, 0) """

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
    objects = getContours()
    if len(objects) >0:
        detection(objects[0])


    key = cv.waitKey(5)
    
    if key == ord("q"):
        break
    elif key == ord("s"):
        break        

cap.release()
cv.destroyAllWindows()
