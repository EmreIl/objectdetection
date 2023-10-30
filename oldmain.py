import cv2 as cv 
import numpy as np
import sys
from matplotlib import pyplot as plt
import classifyObjects as co


capture = cv.VideoCapture(0)

if not capture.isOpened():
    print(f"cannont open camera", file = sys.stderr)
    sys.exit(10)

while True:
    frameAvailable, frame = capture.read()

    if not frameAvailable:
        print(f"no frame availabe", file = sys.stderr)
        break

    cv.imshow("blueObjects", blueLegos) 
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    gray_frame = cv.cvtColor(hsv, cv.COLOR_BGR2GRAY)
    blur_frame = cv.GaussianBlur(gray_frame, (3,3), 0)

    ret, thresh = cv.threshold(blur_frame, 150, 255, cv.THRESH_BINARY)
    
    edges = cv.Canny(blur_frame,100, 200) 

#    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(image=edges, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    cv.drawContours(frame, contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    cv.imshow("Hauptblid", frame)

    blue_objects = []

    for contour in contours:
        min_contour_area = 40 # Passen Sie diesen Wert an

        if cv.contourArea(contour) > min_contour_area:
            x, y, w, h = cv.boundingRect(contour)
            blue_objects.append((x, y, w, h))

    if len(blue_objects) > 0:
        for x, y, w, h in blue_objects:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#    cv.imshow("Original Image with Rectangles", frame)
    key = cv.waitKey(5)
    
    if key == ord("q"):
        break
    elif key == ord("s"):
        break
    elif key == ord("w"): # bei w wird das aktuelle Frame in carFrame.png gespeichert
        outfilepath = "/Users/EM/Fhdw/algorithmen/cameraFrame.png"
        cv.imwrite(outfilepath, blue_lego)
        print(f"frame saved in {outfilepath}")

capture.release()
cv.waitKey(0)
cv.destroyAllWindows()
