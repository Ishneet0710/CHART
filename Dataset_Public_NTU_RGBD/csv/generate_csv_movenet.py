import cv2
import numpy as np
import csv
import tensorflow as tf
import os

# Load MoveNet Model
interpreter = tf.lite.Interpreter(model_path="../../lite-model_movenet_singlepose_lightning_3.tflite")
interpreter.allocate_tensors()

file_header = []
file_header.insert(0, 'class')
for i in range(1, 18):
    file_header.insert(3*i - 2, f'y{i}')
    file_header.insert(3*i - 1, f'x{i}')
    file_header.insert(3*i,     f'c{i}')


def make_csv_files():
    paths = ["./NTURGBD_Training.csv", "./NTURGBD_Test.csv"]
    for pth in paths:
        if not os.path.exists(pth):
            with open(pth, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(file_header)


def make_csv_from_video(video_path, file_path):
    cap = cv2.VideoCapture(video_path)

    if "standing" in video_path:
        pose_class = 0
    elif "sitting" in video_path:
        pose_class = 1

    while cap.isOpened():
        ret, frame = cap.read() # BGR

        if not ret:
            break

        frame = frame[:,:,::-1] # RGB

        img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192,192)
        input_image = tf.cast(img, dtype=tf.float32)

        # Setup input and output 
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make predictions 
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        try:
            for item in keypoints_with_scores:
                pose_row = []
                kp_lst = item[0]
                for kp in kp_lst:
                    y, x, conf = kp
                    pose_row.append(y)
                    pose_row.append(x)
                    pose_row.append(conf)
                pose_row.insert(0, pose_class)

                # Write into csv file
                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(pose_row)
        except:
            pass


def generate_csv(which_csv):
    if which_csv == "Training":
        csv_file_path = "./NTURGBD_Training.csv"
        videos = ['../training/nturgbd_training_standing.mp4', '../training/nturgbd_training_sitting.mp4']
    else:
        csv_file_path = "./NTURGBD_Test.csv"
        videos = ['../test/nturgbd_test_standing.mp4', '../test/nturgbd_test_sitting.mp4']

    for video in videos:
        print(f"Current video: {video}")
        make_csv_from_video(video, csv_file_path)


def main():
    print('-'*75)
    print("MoveNet CSV File Generator")
    print("Options:\n\tA: Training Set\n\tB: Test Set")
    print('-'*75)
    print("Enter choice: ")

    while True:
        inp = input()
        if inp not in ['A', 'B']:
            print("Invalid Input")
            continue
        break

    make_csv_files()
    if inp == 'A':
        generate_csv("Training")
    else:
        generate_csv("Test")

    print("Done!")
    print('-'*75)


if __name__ == '__main__':
    main()