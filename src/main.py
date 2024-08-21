import cv2 as cv
import tensorflow as tf
import numpy as np
from coordinate_system import CoordinateSystem
from detector import Detector
from color_palatte import ColorPalette as cp

labels = []
legend = []

# Create a legend which shows all options
def create_legend():
    global text
    legend.append("s show coordinate system")
    legend.append("a show all legos")
    legend.append("b show 2x6 blue lego")
    legend.append("l show 4x6 black lego")
    legend.append("p disable all options")
    legend.append("1 show all blue legos")
    legend.append("2 show all orange legos")

# loading the keras model and the labels
def load_model():
    global labels
    model_path ="data/main_model/keras_model.h5"
    model = tf.keras.models.load_model(model_path)
    lable_path ="data/main_model/labels.txt"
    with open(lable_path, "r") as file:
        labels = file.read().strip().split('\n')
    return model

def print_labels(frame):
    global legend
    text_start_position = (frame.shape[1] - 300, 50)
    for i, text in enumerate(legend):
        # Calculate position for the text
        text_position = (text_start_position[0], text_start_position[1] + i * 50)
        cv.putText(frame, text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.imshow("Main Frame", frame)

    
def run():
    global labels
    global legend

    cs = CoordinateSystem()
    dt = Detector()
    model = load_model()

    create_legend()
    
    # set all options first disabled
    lego_object = []
    blueLego = False
    findAll = False
    find2x6Blue = False
    find4x6Black = False
    orangeLego = False

    cap = cv.VideoCapture(1)
    cap.set(3, 1280)
    """cap.set(cv.CAP_PROP_AUTOFOCUS, 0) """

    if not cap.isOpened():
        print("Cannot open camera")
        exit(1)

    while True:
        ret, frame = cap.read()
        cv.imshow("Main Frame", frame)
        print_labels(frame)

        if not ret:
            print("No frame available")
            break

        cs.create(frame)

        # actove options
        if blueLego:
            find_legos_by_color(dt, cs, frame,model, cp.BLUE)
        if orangeLego:
            find_legos_by_color(dt, cs, frame,model, cp.ORANGE)
        if findAll:
            locate_all_legos(dt,cs,frame,model)
        if find2x6Blue:
            locate_specfic_Legos(dt,cs,frame,model, cp.BLUE, 3)
        if find4x6Black:
            locate_specfic_Legos(dt,cs,frame,model, cp.BLACK, 5)
        

        key = cv.waitKey(5)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cs.buildFigure(frame) 
        elif key == ord('p'):
            blueLego = False
            findAll = False
            find2x6Blue = False
            find4x6Black = False
            orangeLego = False
        elif key == ord('1'):
            blueLego = True
        elif key == ord('2'):
            orangeLego = True
        elif key == ord('a'):
            findAll = True
        elif key == ord('b'):
            find2x6Blue = True
        elif key == ord('l'):
            find4x6Black = True

    cap.release()
    cv.destroyAllWindows()


# detects legos by color 
def find_legos_by_color(dt, cs, frame, model, color):
    legos = []
    legos = dt.detect_by_color(frame, color)

    locate(legos, frame, model, dt, cs)

# detects all contours in the frame
def locate_all_legos(dt,cs,frame,model):
    legos = []
    legos = dt.detect_contours(frame)

    locate(legos, frame, model, dt, cs)

# detects specific contours in the frame
def locate_specfic_Legos(dt, cs, frame, model, color, number):
    global labels
    legos = []
    legos = dt.detect_by_color(frame, color)

    locate_specific(legos, frame, model, dt, cs, labels[number])

# locate the position and label of the specific lego
def locate_specific(legos, frame, model, dt, cs, label):
     for lego in legos:
        if lego is not None:
            x, y, w, h = cv.boundingRect(lego)
            min_size_threshold = 90

            # validating for None values and too small values
            if all(coord is not None for coord in (x, y, w, h)) and w >= min_size_threshold and h >= min_size_threshold:
                class_label, confidence = dt.identfiy_object(frame, model, labels, lego)
                if class_label == label:
                    text = class_label
                    lego_coords = cs.calculate(lego)
                    lego_x = None
                    lego_y = None
                    
                    if lego_coords is not None:
                        lego_x, lego_y = lego_coords
                    if lego_x is not None and lego_y is not None:
                        text = f'{class_label} Lego Center: ({lego_x:.2f}, {lego_y:.2f})'
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(frame, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.imshow("Main Frame", frame)

# used to get the positon | universal function for other functions
def locate(legos, frame, model, dt, cs):
    for lego in legos:
        if lego is not None:
            x, y, w, h = cv.boundingRect(lego)
            min_size_threshold = 90

            if all(coord is not None for coord in (x, y, w, h)) and w >= min_size_threshold and h >= min_size_threshold:
                class_label, confidence = dt.identfiy_object(frame, model, labels, lego)
                text = class_label
                lego_coords = cs.calculate(lego)
                lego_x = None
                lego_y = None
                
                if lego_coords is not None:
                    lego_x, lego_y = lego_coords
                if lego_x is not None and lego_y is not None:
                    text = f'{class_label} Lego Center: ({lego_x:.2f}, {lego_y:.2f})'
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.imshow("Main Frame", frame)
    

if __name__ == '__main__':
    try:
        run()
    except:
        print('error')