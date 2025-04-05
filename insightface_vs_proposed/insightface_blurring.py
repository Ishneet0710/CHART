import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Change the following lines as needed
FILE_NAME = 'INSIGHTFACE'
ORIGINAL = f"./20230226_222025.mp4"
PROCESSED = f"./{FILE_NAME}_blurred_updated.avi"

cap = cv2.VideoCapture(ORIGINAL)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter(PROCESSED, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

while cap.isOpened():
    ret, img = cap.read()   # BGR
    if not ret:
        break

    try:
        img = img[:,:,::-1] # RGB for InsightFace
        faces = app.get(img)
        img = img[:,:,::-1] # BGR for OpenCV

        for face in faces:
            # Blurring
            x1  = int(face['bbox'][0])
            y1  = int(face['bbox'][1])
            x2 = int(face['bbox'][2])
            y2 = int(face['bbox'][3])

            roi = img[y1:y2, x1:x2]
            roi = cv2.GaussianBlur(roi, (23, 23), 30)
            img[y1:y2, x1:x2] = roi

            result.write(img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    except:
        result.write(img)
        pass

cap.release()
result.release()

print("Done")