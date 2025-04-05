import cv2
import csv
import os
from os.path import isfile, join
from overlap_utils import get_overlap_metrics


# Fall Detection
print("="*50)
print("Fall Detection".center(50))
print("="*50)

proposed_overlap_fall_det = []
insightface_overlap_fall_det = []

trg_dsets = [489, 569, 581, 722, 731, 758, 807, 1219, 1260, 1301, 1373, 1378, 1392, 1790, 1843, 1954]
val_dsets = [1176, 2123]
tst_dsets = [786, 832, 925]

all_dsets = trg_dsets + val_dsets + tst_dsets

for dataset in all_dsets:
    print("Current dataset:", dataset)

    # Get labels
    with open(f"../Dataset_Public_Fall_Detection/images/Dataset {dataset}/Labels/labels.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = [row for row in reader]

    for temp1, temp2 in labels:
        img = int(temp1)
        pose_class = int(temp2)
        if pose_class not in [1, 2, 3]: # Not standing, sitting or lying
            continue
        img_path = f"../Dataset_Public_Fall_Detection/images/Dataset {dataset}/Images/rgb_{img:04d}.png"

        frame = cv2.imread(img_path)    # BGR
        frame = frame[:,:,::-1]         # RGB

        overlap_wrt_proposed, overlap_wrt_insightface = get_overlap_metrics(frame)

        if overlap_wrt_insightface != -1:
            proposed_overlap_fall_det.append(overlap_wrt_proposed)
            insightface_overlap_fall_det.append(overlap_wrt_insightface)

print("Metrics for Fall Detection:")
print(f'{"Overlap / Proposed" : <22}: {sum(proposed_overlap_fall_det) / len(proposed_overlap_fall_det)}')
print(f'{"Overlap / InsightFace" : <22}: {sum(insightface_overlap_fall_det) / len(insightface_overlap_fall_det)}')


# URFall
print("="*50)
print("URFall".center(50))
print("="*50)

proposed_overlap_urfall = []
insightface_overlap_urfall = []

poses = ["standing", "sitting", "lying"]

for temp in ["training", "testing"]:
    print("Current Dataset:", temp.title())
    for pose_class in range(3):
        print(poses[pose_class].title())
        image_directory = f"../Dataset_Public_URFall/{temp}/{poses[pose_class]}"

        images = [join(image_directory, f) for f in os.listdir(image_directory) if (isfile(join(image_directory, f)) and f.endswith('.png'))]

        for img_path in images:
            # Reshape image for processing
            frame = cv2.imread(img_path)    # BGR
            frame = frame[:,:,::-1]         # RGB

            overlap_wrt_proposed, overlap_wrt_insightface = get_overlap_metrics(frame)

            if overlap_wrt_insightface != -1:
                proposed_overlap_urfall.append(overlap_wrt_proposed)
                insightface_overlap_urfall.append(overlap_wrt_insightface)

print("Metrics for URFall:")
print(f'{"Overlap / Proposed" : <22}: {sum(proposed_overlap_urfall) / len(proposed_overlap_urfall)}')
print(f'{"Overlap / InsightFace" : <22}: {sum(insightface_overlap_urfall) / len(insightface_overlap_urfall)}')


# NTU RGB+D
print("="*50)
print("NTU RGB+D".center(50))
print("="*50)

proposed_overlap_nturgbd = []
insightface_overlap_nturgbd = []

poses = ['standing', 'sitting']

for temp in ['training', 'test']:
    for pose in poses:
        print("Current video:", pose, temp)

        video_path = f'../Dataset_Public_NTU_RGBD/{temp}/nturgbd_{temp}_{pose}.mp4'

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read() # BGR

            if not ret:
                break

            frame = frame[:,:,::-1] # RGB

            overlap_wrt_proposed, overlap_wrt_insightface = get_overlap_metrics(frame)

            if overlap_wrt_insightface != -1:
                proposed_overlap_nturgbd.append(overlap_wrt_proposed)
                insightface_overlap_nturgbd.append(overlap_wrt_insightface)

print("Metrics for NTU RGB+D:")
print(f'{"Overlap / Proposed" : <22}: {sum(proposed_overlap_nturgbd) / len(proposed_overlap_nturgbd)}')
print(f'{"Overlap / InsightFace" : <22}: {sum(insightface_overlap_nturgbd) / len(insightface_overlap_nturgbd)}')

print('='*50)
print("FINAL METRICS")
print('='*50)

proposed_overlap = proposed_overlap_fall_det + proposed_overlap_urfall + proposed_overlap_nturgbd
insightface_overlap = insightface_overlap_fall_det + insightface_overlap_urfall + insightface_overlap_nturgbd

print(f'{"Overlap / Proposed" : <22}: {sum(proposed_overlap) / len(proposed_overlap)}')
print(f'{"Overlap / InsightFace" : <22}: {sum(insightface_overlap) / len(insightface_overlap)}')


print('\nDone!')