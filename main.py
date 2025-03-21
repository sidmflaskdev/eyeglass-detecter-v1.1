import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)  # Increased confidence

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Face & Glasses Detection", cv2.WINDOW_NORMAL)

glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")  # Debugging statement
        break

    # Convert frame to RGB (Mediapipe works with RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for face detection
    results = face_detection.process(rgb_frame)

    # Draw face bounding boxes
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Extract the face ROI safely
            x, y, w_box, h_box = max(0, x), max(0, y), min(w, w_box), min(h, h_box)
            face_roi = frame[y:y+h_box, x:x+w_box]
            if face_roi.size == 0:
                continue

            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Use OpenCV's built-in glasses detector with adjusted parameters
            glasses = glasses_cascade.detectMultiScale(gray_face, scaleFactor=1.2, minNeighbors=3, minSize=(20, 20))

            # Determine authorization status
            if len(glasses) > 0:
                status = "Not Authorize"
                color = (0, 0, 255)  # Red
            else:
                status = "Authorize"
                color = (0, 255, 0)  # Green

            # Draw rectangle around face and display status
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the frame
    cv2.imshow('Face & Glasses Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
