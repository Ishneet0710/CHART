import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from Face_Detection_Utils import blur_face
from MoveNet_Drawing_Utils import draw_skeleton

def get_affine_transform_to_fixed_sizes_with_padding(size, new_sizes):
    width, height = new_sizes
    scale = min(height / float(size[1]), width / float(size[0]))
    M = np.float32([[scale, 0, 0], [0, scale, 0]])
    M[0][2] = (width - scale * size[0]) / 2
    M[1][2] = (height - scale * size[1]) / 2
    return M

# Change the following lines as needed
FILE_NAME = 'PROPOSED_NO_SKELETON'
ORIGINAL = f"./20230226_222025.mp4"
PROCESSED = f"./{FILE_NAME}_blurred.avi"

def load_movenet():
    interpreter = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    return interpreter.signatures['serving_default']
MOVENET = load_movenet()

cap = cv2.VideoCapture(ORIGINAL)

width = int(cap.get(3))
height = int(cap.get(4))
size = (width, height)

result = cv2.VideoWriter(PROCESSED, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

while cap.isOpened():
    ret, frame = cap.read()   # BGR
    if not ret:
        break

    try:
        # Reshape image for processing
        img = frame.copy()
        height, width = frame.shape[:2]

        # Variables for drawing MoveNet skeleton and Classifying box
        h_box = np.sqrt(0.0036854 * width * height)
        w_box = 1.7765 * h_box
        x_0 = w_box / 2
        y_0 = h_box / 2

        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
        input_image = tf.cast(img, dtype=tf.int32)
        input_image = input_image[:,:,::-1] # RGB for MoveNet

        # MoveNet detections
        res = MOVENET(input_image)
        keypoints_with_scores = res['output_0'].numpy()[:,:,:51].reshape((6,17,3))

        for i in range(len(keypoints_with_scores)):
            if i >= 1:
                break
            # Note: person is normalised keypoints, but keypoints_with_scores represent the actual coordinates on the frame
            person = keypoints_with_scores[i]

            kp_with_scores = person.copy()    # For drawing skeleton                
            M = get_affine_transform_to_fixed_sizes_with_padding((height, width), (192, 192))
            M = np.vstack((M, [0, 0, 1]))
            M_inv = np.linalg.inv(M)[:2]
            xy_keypoints = kp_with_scores[:, :2] * 192
            xy_keypoints = cv2.transform(np.array([xy_keypoints]), M_inv)[0]
            kp_with_scores = np.hstack((xy_keypoints, kp_with_scores[:, 2:]))
            coords = kp_with_scores.flatten()

            frame = np.ascontiguousarray(frame, dtype=np.uint8) # Resolve errors in drawing

            # Flip image horizontally, fixes an error which idk why it happens -_-
            frame = cv2.flip(frame, 1)

            # Rendering
            blur_face(frame, coords)
            # draw_skeleton(frame, kp_with_scores, 0, x_0)

            frame = cv2.flip(frame, 1)

            result.write(frame)

    except Exception as e:
        print(e)
        pass

cap.release()
result.release()

print("Done")