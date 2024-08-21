import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt

class CoordinateSystem(object):

    yellow_circle = None
    yellow_center = None
    
    red_circle = None
    red_center = None

    x_length = None
    y_length = None

    # finding the circles to create an coordinate system
    def create(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_blurred = cv.GaussianBlur(gray, (9, 9), 2)

        # identify the circles 
        circles = cv.HoughCircles(
            gray_blurred,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )

        red_lower = np.array([160, 100, 100])
        red_upper = np.array([180, 255, 255])

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])

        if circles is not None:

            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                
                roi = frame[y - r:y + r, x - r:x + r]
                if roi is not None:
                    hsv_image = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

                    yellow_mask = cv.inRange(hsv_image, yellow_lower, yellow_upper)
                    red_mask = cv.inRange(hsv_image, red_lower, red_upper)

                    # if the circle is red than its the 1/1 and when yellow the 0/0 positon
                    if np.any(yellow_mask > 0):
                        self.yellow_circle = (x,y,r)
                    if np.any(red_mask > 0):
                        self.red_circle = (x,y,r)

                if self.yellow_circle is not None and self.red_circle is not None:
                    self.yellow_center = self.yellow_circle[:2]
                    self.red_center = self.red_circle[:2]
                    self.x_length = self.red_center[0] - self.yellow_center[0]
                    self.y_length = self.yellow_center[1] - self.red_center[1]

    # using the coordinates from the circles and the lego to calculate the position
    def calculate(self, countour):

        x, y, w, h = cv.boundingRect(countour)
        if all(coord is not None for coord in (x, y, w, h,self.yellow_center[0],self.yellow_center[1])):
            lego_contour = countour

            # finding the center of the lego
            M = cv.moments(lego_contour)
            if M["m00"] != 0:  
                lego_center_x = int(M["m10"] / M["m00"])
                lego_center_y = int(M["m01"] / M["m00"])
                lego_center = (lego_center_x, lego_center_y)

                # convert the frame coordinates to the new coordinate system between 0/0 and 
                normalized_lego_x = (lego_center_x - self.yellow_center[0]) / self.x_length
                normalized_lego_y = (self.yellow_center[1] - lego_center_y) / self.y_length

                normalized_lego_x = max(0, min(normalized_lego_x, 1))
                normalized_lego_y = max(0, min(normalized_lego_y, 1))

                if normalized_lego_x != 0 and normalized_lego_y !=0:
                    return (normalized_lego_x, normalized_lego_y)
            
    # Building a figure to check the values of the calculated position
    def buildFigure(self, frame):
        fig, ax = plt.subplots()

        ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        ax.scatter(*self.yellow_center, color='yellow', label='Yellow Circle')
        ax.scatter(*self.red_center, color='red', label='Red Circle')

        ax.plot([self.red_center[0], self.yellow_center[0]], [self.yellow_center[1], self.yellow_center[1]], 
                color='blue', linestyle='--', label='X Line')
        ax.plot([self.yellow_center[0], self.yellow_center[0]], [self.yellow_center[1], self.red_center[1]], 
                color='green', linestyle='--', label='Y Line')

        ax.set_aspect('equal', adjustable='box')

        x_ticks = np.linspace(self.yellow_center[0], self.red_center[0], 5)
        y_ticks = np.linspace(self.yellow_center[1], self.red_center[1], 5)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        ax.set_xticklabels([f'{(val - self.yellow_center[0]) / (self.red_center[0] - self.yellow_center[0]):.2f}' for val in x_ticks])
        ax.set_yticklabels([f'{(val - self.yellow_center[1]) / (self.red_center[1] - self.yellow_center[1]):.2f}' for val in y_ticks])

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        ax.legend()

        plt.grid(True)
        plt.show()