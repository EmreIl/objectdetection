import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('pictures/koordniatenBlatt4.jpeg')

cap = cv2.VideoCapture(1)
cap.set(3, 1280)

if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("No frame available")
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
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
            
            roi = image[y - r:y + r, x - r:x + r]
            hsv_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            """ red_lower = np.array([0, 100, 100])
            red_upper = np.array([10, 255, 255]) """

            red_lower = np.array([160, 100, 100])
            red_upper = np.array([180, 255, 255])

            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])

            red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
            yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

            if np.any(yellow_mask > 0):
                yellow_circle = (x,y,r)
            if np.any(red_mask > 0):
                red_circle = (x,y,r)

            """ yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            yellow_circle = yellow_contours[0]
            red_circle = red_contours[0] """

            # Get the center coordinates of the circles
            """ yellow_center = tuple(yellow_circle[:, 0, :].mean(axis=0).astype(int))
            red_center = tuple(red_circle[:, 0, :].mean(axis=0).astype(int)) """

            

            """ midpoint = (yellow_center[0] + red_center[0]) // 2, (yellow_center[1] + red_center[1]) // 2
            secondpoint = (yellow_center[0], (yellow_center[1] + red_center[1]) // 2)
            normalized_x = (yellow_center[0] - midpoint[0]) / x_length
            normalized_y = (midpoint[1] - red_center[1]) / y_length """

        if yellow_circle is not None and red_circle is not None:
            print("testtt")
            yellow_center = yellow_circle[:2]
            red_center = red_circle[:2]
            x_length = red_center[0] - yellow_center[0]
            y_length = yellow_center[1] - red_center[1]
        else:
            print("Failed")
            
        cv2.imshow("Frame", frame)
        

    key = cv2.waitKey(5)
    if key == ord("q"):
        break
    if key == ord("s"):
        # Set up the figure
        fig, ax = plt.subplots()

        # Plot the circles and coordinate system
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
           

cap.release()
cv2.destroyAllWindows()




    #print(f"Normalized Coordinates: ({normalized_x:.2f}, {normalized_y:.2f})")


""" lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lego_contour = blue_contours[0]

    # Calculate the centroid of the Lego contour
    M = cv2.moments(lego_contour)
    lego_center_x = int(M["m10"] / M["m00"])
    lego_center_y = int(M["m01"] / M["m00"])
    lego_center = (lego_center_x, lego_center_y)

    # Assuming 'yellow_center' and 'red_center' are the centers of the yellow and red circles
    # Calculate the lengths of x and y lines
    x_length = red_center[0] - yellow_center[0]
    y_length = yellow_center[1] - red_center[1]

    # Calculate normalized coordinates for the Lego center
    normalized_lego_x = (lego_center_x - yellow_center[0]) / x_length
    normalized_lego_y = (yellow_center[1] - lego_center_y) / y_length

    # Print the coordinates of the Lego center
    print(f"Lego Center Coordinates: {lego_center}")
    print(f"Normalized Coordinates for Lego Center: ({normalized_lego_x:.2f}, {normalized_lego_y:.2f})")

    # Display the result
    cv2.imshow('Detected Circles', image) """


""" # Build a coordinate system based on the circles
    x_range = purple_x - red_x
    y_range = purple_y - red_y


    # Find the black rectangle
    roi = image[red_y:purple_y, red_x:purple_x]
    
    cv2.imshow('dd Circles and Rectangle', roi)

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary mask of the black rectangle
    _, thresholded = cv2.threshold(gray_roi, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(roi, contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    if contours:
        # Find the bounding box of the largest contour (assuming it's the rectangle)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the center of the rectangle
        rect_center_x = x + w // 2
        rect_center_y = y + h // 2

        # Calculate normalized coordinates for the rectangle
        normalized_rect_x = (rect_center_x - red_x) / x_range
        normalized_rect_y = (rect_center_y - red_y) / y_range

        print(f"Rectangle at normalized coordinates: ({normalized_rect_x:.2f}, {normalized_rect_y:.2f})")

        # Draw a bounding box around the rectangle
        cv2.rectangle(image, (red_x + x, red_y + y), (red_x + x + w, red_y + y + h), (0, 255, 0), 2)

    cv2.imshow('Detected Circles and Rectangle', image) """
    