import cv2 as cv 
import numpy as np
import sys
from matplotlib import pyplot as plt


capture = cv.VideoCapture(1)

if not capture.isOpened():
    print(f"cannont open camera", file = sys.stderr)
    sys.exit(10)

while True:
    frameAvailable, frame = capture.read()
    hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    if not frameAvailable:
        print(f"no frame availabe", file = sys.stderr)
        break

    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])

    pink_lower = np.array([160, 100, 100])
    pink_upper = np.array([180, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    """ red_mask = cv.inRange(hsv_image, red_lower, red_upper)
    yellow_mask = cv.inRange(hsv_image, yellow_lower, yellow_upper)
    pink_mask = cv.inRange(hsv_image, pink_lower, pink_upper)

    yellow_contours, _ = cv.findContours(yellow_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv.findContours(pink_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) """


    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_blurred = cv.GaussianBlur(gray, (9, 9), 2)

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

    if circles is not None:

        circles = np.round(circles[0, :]).astype("int")
        yellow_circle = None
        red_circle = None
        yellow_center = None
        red_center = None

        for (x, y, r) in circles:
            
            roi = frame[y - r:y + r, x - r:x + r]
            hsv_image = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

            yellow_mask = cv.inRange(hsv_image, yellow_lower, yellow_upper)
            pink_mask = cv.inRange(hsv_image, pink_lower, pink_upper)

            if np.any(yellow_mask > 0):
                yellow_circle = (x,y,r)
            if np.any(pink_mask > 0):
                red_circle = (x,y,r)


        if yellow_circle is not None and red_circle is not None:
            yellow_center = yellow_circle[:2]
            red_center = red_circle[:2]
            x_length = red_center[0] - yellow_center[0]
            y_length = yellow_center[1] - red_center[1]

            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])

            hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            blue_mask = cv.inRange(hsv_image, lower_blue, upper_blue)
            blue_contours, _ = cv.findContours(blue_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Sort contours by area in descending order
            blue_contours = sorted(blue_contours, key=cv.contourArea, reverse=True)

            # Use the largest contour as the Lego contour
            if blue_contours:
                lego_contour = blue_contours[0]

                # Calculate the moments and center of the Lego contour
                M = cv.moments(lego_contour)
    
                if M["m00"] != 0:  # Check if area is not zero
                    lego_center_x = int(M["m10"] / M["m00"])
                    lego_center_y = int(M["m01"] / M["m00"])
                    lego_center = (lego_center_x, lego_center_y)

                    normalized_lego_x = (lego_center_x - yellow_center[0]) / x_length
                    normalized_lego_y = (yellow_center[1] - lego_center_y) / y_length

                    # Scale the normalized coordinates to be within the range [0, 1]
                    normalized_lego_x = max(0, min(normalized_lego_x, 1))
                    normalized_lego_y = max(0, min(normalized_lego_y, 1))

                    """ # Convert coordinates to match matplotlib's coordinate system
                    normalized_lego_y = 1 - normalized_lego_y """

                    print(f"Lego Center Coordinates: {lego_center}")
                    print(f"Normalized Coordinates for Lego Center: ({normalized_lego_x:.2f}, {normalized_lego_y:.2f})")
                        
        else:
            print("Failed")

    cv.imshow("frametest", frame)

    key = cv.waitKey(5)
    
    if key == ord("q"):
        break
    if key == ord("s"):
        # Set up the figure
        fig, ax = plt.subplots()

        # Plot the circles and coordinate system
        ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        ax.scatter(*yellow_center, color='yellow', label='Yellow Circle')
        ax.scatter(*red_center, color='red', label='Red Circle')

        # Draw x and y lines
        ax.plot([red_center[0], yellow_center[0]], [yellow_center[1], yellow_center[1]], color='blue', linestyle='--', label='X Line')
        ax.plot([yellow_center[0], yellow_center[0]], [yellow_center[1], red_center[1]], color='green', linestyle='--', label='Y Line')

        # Set aspect ratio to 'equal'
        ax.set_aspect('equal', adjustable='box')

        # Determine tick positions and labels
        x_ticks = np.linspace(yellow_center[0], red_center[0], 5)
        y_ticks = np.linspace(yellow_center[1], red_center[1], 5)

        # Set ticks and labels
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        ax.set_xticklabels([f'{(val - yellow_center[0]) / (red_center[0] - yellow_center[0]):.2f}' for val in x_ticks])
        ax.set_yticklabels([f'{(val - yellow_center[1]) / (red_center[1] - yellow_center[1]):.2f}' for val in y_ticks])

        # Add labels
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        # Add legend
        ax.legend()

        # Show the plot
        plt.grid(True)
        plt.show()
           


capture.release()
cv.waitKey(0)
cv.destroyAllWindows()
