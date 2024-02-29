import cv2

# Load the Haar cascade classifier for face detection
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize a variable to keep track of the number of people detected
num_people = 0

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame using the Haar cascade
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    
    # Loop over each face and draw a bounding box around it
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Update the number of people detected
    num_people = len(faces)
    
    # Draw the number of people detected on the frame
    cv2.putText(frame, "Number of people: " + str(num_people), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Wait for 1 millisecond for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
