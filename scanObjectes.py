import cv2 as cv 
import numpy as np
import sys
from matplotlib import pyplot as plt
import os


capture = cv.VideoCapture(0)

if not capture.isOpened():
    print(f"cannont open camera", file = sys.stderr)
    sys.exit(10)

count = 3

outfilepath = "/home/emre/Projekte/objectdetection/data/"
outfilepath += "2x4green/"

filetowrite = "pic" + str(count) + ".jpg"

while True:
    frameAvailable, frame = capture.read()

    if not frameAvailable:
        print(f"no frame availabe", file = sys.stderr)
        break
    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    cv.imshow("Hauptblid", frame)

    key = cv.waitKey(5)
    
    if key == ord("q"):
        break
    elif key == ord("s"):
        break
    elif key == ord("w"): # bei w wird das aktuelle Frame in carFrame.png gespeichert

        if os.path.exists(os.path.join(outfilepath, filetowrite)):
            count+=1
            filetowrite = "pic" + str(count) + ".jpg"
        else: 
            resultfile = outfilepath + filetowrite
            print(resultfile)
            cv.imwrite(resultfile, frame)
            print(f"frame saved in {outfilepath}")
            count +=1
            filetowrite = "pic" + str(count) + ".jpg"

capture.release()
cv.waitKey(0)
cv.destroyAllWindows()
