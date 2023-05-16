import cv2
import numpy as np

# Create a VideoCapture object to capture video from the camera
cap = cv2.VideoCapture(0)

# Loop indefinitely to read frames from the camera
while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the input image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the grayscale image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply thresholding to the blurred image
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find the contours in the thresh held image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour, which is assumed to be the hand
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
        else:
            continue
        finger = max_contour


        # Define the finger contour as the largest contour
        # Compute the convexity defects of the finger contour
        defects = cv2.convexityDefects(finger, cv2.convexHull(finger, returnPoints=False))

        # Check if defects is not None before iterating over its shape
        if defects is not None:

            # Iterate over the defects and count the number of fingers
            num_fingers = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(finger[s][0])
                end = tuple(finger[e][0])
                far = tuple(finger[f][0])
                a = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
                b = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2
                c = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2
                angle = np.arccos((b + c - a) / np.sqrt(4 * b * c)) * 180 / np.pi
                if angle <= 90:
                    num_fingers += 1
        else:
            # Set the number of fingers to zero if there are no convexity defects
            num_fingers = 0

        # Display the number of fingers in the image
        cv2.putText(frame, str(num_fingers), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the processed frame
        cv2.imshow('Finger Counting', frame)

        # Check if the 'q' key is pressed to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

# Release the VideoCapture object and destroy all windows
cap.release()
cv2.destroyAllWindows()


###It depends on the lighting of the room, background of the room.
