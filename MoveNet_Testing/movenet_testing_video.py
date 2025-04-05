import cv2
from MoveNet_Processing_Utils import movenet_processing

# Change this accordingly
cap = cv2.VideoCapture('../Dataset_Own/videos/standing_09.mp4')


while cap.isOpened():
    ret, frame = cap.read() # BGR
    frame = frame[:,:,::-1] # RGB

    processed_frame = movenet_processing(frame) # RGB
    processed_frame = processed_frame[:,:,::-1] # BGR

    cv2.imshow('MoveNet Test', processed_frame)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()