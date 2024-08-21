import cv2 as cv 
import numpy as np

class Detector(object):

    # detect all contours of the frame
    def detect_contours(self, frame):
        objects = []

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur_frame = cv.GaussianBlur(gray_frame, (9,9), 0)
        edges = cv.Canny(blur_frame,100, 200) 

        # validating for better results
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv.dilate(edges, kernel, iterations=1)
        eroded_edges = cv.erode(dilated_edges, kernel, iterations=1)

        contours, hierarchy = cv.findContours(image=eroded_edges, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_TC89_L1)
        #cv.drawContours(frame, contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

        # append all contours to an object array
        for contour in contours:
            min_contour_area = 40 
            if cv.contourArea(contour) > min_contour_area:
                x = y = w = h = None
                x, y, w, h = cv.boundingRect(contour)
                if all(coord is not None for coord in (x, y, w, h)):
                    objects.append(contour)

        if objects != []:
            return objects
        
    # detects by specific color
    def detect_by_color(self, frame, color):
        objects = []

        # process image to get the contours
        hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_image, color['lower'], color['upper'])
        color_contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # validate the contours
        for contour in color_contours:
            min_contour_area = 40 
            if cv.contourArea(contour) > min_contour_area:
                x = y = w = h = None
                x, y, w, h = cv.boundingRect(contour)
                if all(coord is not None for coord in (x, y, w, h)):
                    objects.append(contour)
        
        if objects != []:
            return objects
        
    # using the model to identify the object
    def identfiy_object(self, frame, model, labels, object):
        roi = frame
        confidence_threshold = 0.7

        # cutting the lego out of the full image to identify it 
        x, y, w, h = cv.boundingRect(object)
        roi = frame[y:y+h, x:x+w]

        if roi is not None and roi.any():
            # process it so it can be used in the model.predic
            input_image = cv.resize(roi, (224, 224))
            input_image = input_image / 255.0  

            # identify the object
            predictions = model.predict(np.expand_dims(input_image, axis=0))[0]
            class_label = None
            confidence = None

            for i, confidence in enumerate(predictions):
                if confidence > confidence_threshold:
                    class_label = labels[i]
                    text = f"{class_label}: {confidence:.2f}"

        if class_label != None and confidence != None:
            return (class_label, confidence)
        return ('','')
    