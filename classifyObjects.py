import cv2 as cv 
import numpy as np
import sys
from matplotlib import pyplot as plt
import os

def writeModifiedImageToDir(input_dir, output_dir, lower_color, upper_color):
    if not os.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)
            inputFile = input_dir + "/"+ filename
            newFile = addBoundingBoxesToImage(inputFile, lower_color, upper_color)
            cv.imwrite(output_image_path, newFile)

def addBoundingBoxesToImage(input_image, lower_color, upper_color):

    frame = cv.imread(input_image)
    objects = []

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    blur_frame = cv.GaussianBlur(hsv, (3,3), 0)


    mask = cv.inRange(blur_frame, lower_color, upper_color)

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
    
    return frame

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

input_image_path = "/home/emre/Projekte/objectdetection/data/2x4blue"
output_image_path = "/home/emre/Projekte/objectdetection/data/2x3blueModified/"

def showImage():
    path = input_image_path + "/pic13.jpg"
    f = addBoundingBoxesToImage(path, lower_blue, upper_blue)
    cv.imshow("picture", f)
    cv.waitKey(0)
    cv.destroyAllWindows()


def augmentation():
    import tensorflow as tf

    img = input_image_path + "/pic13.jpg"
# Lade dein Bild
    image = tf.io.read_file(img)
    image = tf.image.decode_image(image)

# Anwendung von Data Augmentation
    image = tf.image.random_brightness(image, max_delta=0.2)  # Helligkeit ändern
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # Kontrast ändern
    image = tf.image.random_flip_left_right(image)  # Horizontal spiegeln
    image = tf.image.random_flip_up_down(image)  # Vertikal spiegeln
    image = tf.image.random_crop(image, size=[200, 200, 3])  # Zufälliges Zuschneiden auf 200x200 Pixel

# Zeige das veränderte Bild (nur für Illustrationszwecke)
    import matplotlib.pyplot as plt
    plt.imshow(image.numpy())
    plt.show()

#augmentation()
showImage()
#writeModifiedImageToDir(input_image_path, output_image_path, lower_blue, upper_blue)

