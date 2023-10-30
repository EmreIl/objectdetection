import cv2 as cv 
import numpy as np
import sys
from matplotlib import pyplot as plt


capture = cv.VideoCapture(0)

if not capture.isOpened():
    print(f"cannont open camera", file = sys.stderr)
    sys.exit(10)

while True:
    frameAvailable, frame = capture.read()

    if not frameAvailable:
        print(f"no frame availabe", file = sys.stderr)
        break

    objects = []

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    blur_frame = cv.GaussianBlur(hsv, (3,3), 0)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv.inRange(blur_frame, lower_blue, upper_blue)

#    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(image=mask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

    for contour in contours:
        min_contour_area = 10 # Passen Sie diesen Wert an

        if cv.contourArea(contour) > min_contour_area:
            x, y, w, h = cv.boundingRect(contour)
            objects.append((x, y, w, h))

    if len(objects) > 0:
        for x, y, w, h in objects:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "2x3 blue lego"  # Ihr gewünschter Text
            text_x = x  # Die X-Koordinate des Textes (z. B. links oben)
            text_y = y + h + 20  # Die Y-Koordinate des Textes (z. B. etwas unter dem Rechteck)
            cv.putText(frame, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imshow("frame", frame)
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
