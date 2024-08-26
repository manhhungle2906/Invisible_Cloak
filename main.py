import cv2
import numpy as np
import time

# Prepare for writing the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Allow the system to sleep for 3 seconds before the webcam starts
time.sleep(3)
count = 0
background = 0

# Capture the background in range of 60 frames
for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis=1)

# Read every frame from the webcam until the camera is open
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    img = np.flip(img, axis=1)
    
    # Convert the color space from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generate masks to detect red color
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask1 = mask1 + mask2

    # Open and Dilate the mask image
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Create an inverse mask
    mask2 = cv2.bitwise_not(mask1)

    # Extract the red regions from the frame
    res1 = cv2.bitwise_and(img, img, mask=mask2)

    # Extract the background regions where red cloak is present
    res2 = cv2.bitwise_and(background, background, mask=mask1)

    # Combine the background and the original image
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Write the frame to the output video
    out.write(final_output)

    # Display the result
    cv2.imshow('Invisible Cloak', final_output)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
