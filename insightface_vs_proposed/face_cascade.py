import cv2

# Change the following lines as needed
FILE_NAME = 'CASCADE'
ORIGINAL = f"./20230226_222025.mp4"
PROCESSED = f"./{FILE_NAME}_blurred_updated.avi"

cap = cv2.VideoCapture(ORIGINAL)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter(PROCESSED, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
while cap.isOpened():
    ret, img = cap.read()   # BGR
    if not ret:
        break

    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(img_gray, 1.1, 9)

        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
            roi = cv2.GaussianBlur(roi, (23, 23), 30)
            img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

        result.write(img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    except:
        result.write(img)
        pass

cap.release()
result.release()

print("Done")