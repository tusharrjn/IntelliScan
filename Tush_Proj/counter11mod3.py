import cv2
import dlib
import math
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Initialize the face detector and the points predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Initialize the arrays to store the posture data
head_tilt_degrees_arr = []
upper_lip_distances_arr = []
lower_lip_distances_arr = []
yawning_distances_arr = []

# Initialize the threshold values for yawning and head tilt
yawning_thresh = 50
head_tilt_thresh = 15

# Initialize the decision tree regressor
regressor = DecisionTreeRegressor()

while True:
    # Read the frame from the video capture object
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the frame
    faces = detector(gray)

    # Loop through each face and calculate the yawning and head tilt
    for face in faces:
        landmarks = predictor(gray, face)
        mouth_left = (landmarks.part(60).x, landmarks.part(60).y)
        mouth_right = (landmarks.part(64).x, landmarks.part(64).y)
        mouth_center = (landmarks.part(62).x, landmarks.part(62).y)
        upper_lip_distance = math.sqrt((mouth_center[0] - mouth_left[0]) ** 2 + (mouth_center[1] - mouth_left[1]) ** 2)
        lower_lip_distance = math.sqrt((mouth_center[0] - mouth_right[0]) ** 2 + (mouth_center[1] - mouth_right[1]) ** 2)
        yawning_distance = upper_lip_distance + lower_lip_distance
        head_tilt = abs(landmarks.part(17).y - landmarks.part(26).y)
        head_tilt_degrees = math.degrees(math.atan(head_tilt / yawning_distance))

        # Append the posture data to the arrays
        head_tilt_degrees_arr.append(head_tilt_degrees)
        upper_lip_distances_arr.append(upper_lip_distance)
        lower_lip_distances_arr.append(lower_lip_distance)
        yawning_distances_arr.append(yawning_distance)

        # Check if we have enough data to train the model
        if len(head_tilt_degrees_arr) > 10:
            # Train the decision tree regressor on the posture data
            X = np.column_stack((head_tilt_degrees_arr, upper_lip_distances_arr, lower_lip_distances_arr, yawning_distances_arr))
            y = np.array([head_tilt_thresh]*len(head_tilt_degrees_arr))
            regressor.fit(X, y)

            # Use the decision tree regressor to adjust the threshold values for yawning and head tilt
            X_pred = np.array([[head_tilt_degrees_arr[-1], upper_lip_distances_arr[-1], lower_lip_distances_arr[-1], yawning_distances_arr[-1]]])
            head_tilt_thresh = int(regressor.predict(X_pred)[0])
            yawning_thresh = int(np.mean(yawning_distances_arr) + 2 * np.std(yawning_distances_arr))

        # Check if the person is yawning or tilting their head
        if yawning_distance > yawning_thresh or head_tilt_degrees > head_tilt_thresh:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        # Display the head tilt degree and upper lip and lower lip distance in the live feed
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Head Tilt Degree: {head_tilt_degrees:.2f}", (face.left(), face.top() - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Upper Lip Distance: {upper_lip_distance:.2f}", (face.left(), face.top() + 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Lower Lip Distance: {lower_lip_distance:.2f}", (face.left(), face.top() + 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Head Tilt Threshold: {head_tilt_thresh:.2f}", (10, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Yawning Threshold: {yawning_thresh:.2f}", (10, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame on the screen
    cv2.imshow("Live Feed", frame)

    # Stop the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

