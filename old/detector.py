import cv2 as cv 
import numpy as np

class Detector(object):

    objects = []

    def detect_contours(self, frame):
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur_frame = cv.GaussianBlur(gray_frame, (9,9), 0)
        edges = cv.Canny(blur_frame,100, 200) 

        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv.dilate(edges, kernel, iterations=1)
        eroded_edges = cv.erode(dilated_edges, kernel, iterations=1)

        contours, hierarchy = cv.findContours(image=eroded_edges, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_TC89_L1)
        cv.drawContours(frame, contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

        for contour in contours:
            min_contour_area = 40 # Passen Sie diesen Wert an
            if cv.contourArea(contour) > min_contour_area:
                x = y = w = h = None
                x, y, w, h = cv.boundingRect(contour)
                if all(coord is not None for coord in (x, y, w, h)):
                    self.objects.append((x, y, w, h))

        if self.objects != []:
            return self.objects
        
    def identfiy_object(self, frame, model, labels, object):
        offset = 5  
        roi = frame
        confidence_threshold = 0.7
        if len(self.objects) != 0:
            
            x,y,w,h = self.objects
            """ x -= offset
            y -= offset
            w += 2 * offset
            h += 2 * offset """
            roi = frame[y:y+h, x:x+w]

        if roi is not None and roi.any():
            input_image = cv.resize(roi, (224, 224))
            input_image = input_image / 255.0  

            predictions = model.predict(np.expand_dims(input_image, axis=0))[0]

            for i, confidence in enumerate(predictions):
                if confidence > confidence_threshold:
                    class_label = labels[i]
                    text = f"{class_label}: {confidence:.2f}"
                    
