import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def load_movenet():
    interpreter = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    return interpreter.signatures['serving_default']

def load_insightface():
    app = FaceAnalysis(allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

MOVENET = load_movenet()
INSIGHTFACE = load_insightface()

def get_affine_transform_to_fixed_sizes_with_padding(size, new_sizes):
    width, height = new_sizes
    scale = min(height / float(size[1]), width / float(size[0]))
    M = np.float32([[scale, 0, 0], [0, scale, 0]])
    M[0][2] = (width - scale * size[0]) / 2
    M[1][2] = (height - scale * size[1]) / 2
    return M

def proposed_method_bbox(coords):
    keypoints_to_consider = [0, 1, 2, 3, 4, 5, 6, 11, 12]
    y_coords = [coords[::3][idx] for idx in keypoints_to_consider]
    x_coords = [coords[1::3][idx] for idx in keypoints_to_consider]

    x_body_range = max(x_coords) - min(x_coords)
    y_body_range = max(y_coords) - min(y_coords)
    ratio = x_body_range / y_body_range

    x_face = x_coords[:5]
    y_face = y_coords[:5]
    x_bar = np.mean(x_coords[:5])
    y_bar = np.mean(y_coords[:5])

    # Get largest |x_i - x_bar|
    x_max = -1
    for x_i in x_face:
        temp = abs(x_i - x_bar)
        if temp > x_max:
            x_max = temp

    # Get largest |y_i - y_bar|
    y_max = -1
    for y_i in y_face:
        temp = abs(y_i - y_bar)
        if temp > y_max:
            y_max = temp

    if ratio > 1:
        h_head = 2.5 * y_max
        w_head = 1.5 * h_head
    else:
        w_head = 2.5 * x_max
        h_head = 1.5 * w_head

    return int(x_bar - 0.5*w_head), int(y_bar - 0.5*h_head), int(x_bar + 0.5*w_head), int(y_bar + 0.5*h_head)

def overlap(a, b, c, d):
    '''
    Intersection of the intervals [a, b] and [c, d]
    '''
    if b <= c or d <= a:
        return 0
    else:
        temp = sorted([a, b, c, d])
        return temp[2] - temp[1]

def get_overlap_metrics(frame):
    '''
    FIRST - Proposed heuristic method
            Assume only 1 face detected
    '''
    height, width = frame.shape[:2]

    # Reshape image for processing
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.int32)

    # MoveNet detections
    res = MOVENET(input_image)
    keypoints_with_scores = res['output_0'].numpy()[:,:,:51].reshape((6,17,3))

    person = keypoints_with_scores[0]

    kp_with_scores = person.copy()              
    M = get_affine_transform_to_fixed_sizes_with_padding((height, width), (192, 192))
    M = np.vstack((M, [0, 0, 1]))
    M_inv = np.linalg.inv(M)[:2]
    xy_keypoints = kp_with_scores[:, :2] * 192
    xy_keypoints = cv2.transform(np.array([xy_keypoints]), M_inv)[0]
    kp_with_scores = np.hstack((xy_keypoints, kp_with_scores[:, 2:]))
    coords = kp_with_scores.flatten()

    x1_proposed, y1_proposed, x2_proposed, y2_proposed = proposed_method_bbox(coords)

    '''
    SECOND - InsightFace method
    '''
    faces = INSIGHTFACE.get(frame)

    try:
        face_to_consider = faces[0]
        x1_insightface  = int(face_to_consider['bbox'][0])
        y1_insightface  = int(face_to_consider['bbox'][1])
        x2_insightface = int(face_to_consider['bbox'][2])
        y2_insightface = int(face_to_consider['bbox'][3])
    except:
        return -1, -1

    area_overlap = overlap(x1_proposed, x2_proposed, x1_insightface, x2_insightface) * overlap(y1_proposed, y2_proposed, y1_insightface, y2_insightface)
    area_proposed = abs((x2_proposed - x1_proposed) * (y2_proposed - y1_proposed))
    area_insightface = abs((x2_insightface - x1_insightface) * (y2_insightface - y1_insightface))

    overlap_wrt_proposed = area_overlap / area_proposed
    overlap_wrt_insightface = area_overlap / area_insightface

    return overlap_wrt_proposed, overlap_wrt_insightface