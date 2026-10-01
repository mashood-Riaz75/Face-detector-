import cv2

# Load the face detector
face_detector = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

# Read image
image = cv2.imread("images/test.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

# Draw boxes around detected faces
for (x, y, w, h) in faces:

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        "Face",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# Show result
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()