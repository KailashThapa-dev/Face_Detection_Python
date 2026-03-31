import cv2
# Load the Haar Cascade classifier for face detection
face_capture = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Enable camera
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Check if the camera opened successfully
while True:
    ret, video_data = video_capture.read()
    # Convert the video frame to grayscale for face detection
    color = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
    # Detect faces in the video frame   
    faces = face_capture.detectMultiScale(color,
                                 scaleFactor=1.1,
                                 minNeighbors=5,
                                 minSize=(30, 30),
                                 flags=cv2.CASCADE_SCALE_IMAGE
                                 )
    # Draw rectangles around the detected faces
    for(x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
# Check if the frame was captured successfully
    if not ret or video_data is None:
        print("Failed to grab video frame")
        break
# Display the video feed
    cv2.imshow("video_live", video_data)

    # Press 'a' (small a) to exit
    if cv2.waitKey(10) & 0xFF == ord('a'):
        break


video_capture.release()
cv2.destroyAllWindows()