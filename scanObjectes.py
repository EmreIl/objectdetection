import cv2 as cv 
import numpy as np
import sys
from matplotlib import pyplot as plt
import os


capture = cv.VideoCapture(1)

if not capture.isOpened():
    print(f"cannont open camera", file = sys.stderr)
    sys.exit(10)

count = 0



filetowrite = "pic" + str(count) + ".jpg"

while True:
    frameAvailable, frame = capture.read()

    if not frameAvailable:
        print(f"no frame availabe", file = sys.stderr)
        break
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur_frame = cv.GaussianBlur(gray_frame, (9,9), 0)
    edges = cv.Canny(blur_frame,50, 150) 

    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv.dilate(edges, kernel, iterations=1)
    eroded_edges = cv.erode(dilated_edges, kernel, iterations=1)

#    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(image=eroded_edges, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_TC89_L1)

    x, y, w, h = cv.boundingRect(contours[0])
    roi = frame[y:y+h, x:x+w]

    cv.imshow('Rotated ROI', roi)

    key = cv.waitKey(5)
    
    if key == ord("q"):
        break
    elif key == ord("s"):
        break
    elif key == ord("w"): # bei w wird das aktuelle Frame in carFrame.png gespeichert
        print("test")
        outfilepath = "data/2x3blue/"
        if os.path.exists(os.path.join(outfilepath, filetowrite)):
            count+=1
            filetowrite = "pic" + str(count) + ".jpg"
        else: 
            resultfile = outfilepath + filetowrite
            print(resultfile)
            cv.imwrite(resultfile, roi)
            print(f"frame saved in {outfilepath}")
            count +=1
            filetowrite = "pic" + str(count) + ".jpg"

capture.release()
cv.waitKey(0)
cv.destroyAllWindows()
