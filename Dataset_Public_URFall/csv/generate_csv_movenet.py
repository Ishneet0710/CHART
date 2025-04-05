import cv2
import numpy as np
import csv
import os
from os.path import isfile, join
import tensorflow as tf

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
    paths = ["./Public_URFall_Training.csv", "./Public_URFall_Test.csv"]
    for pth in paths:
        if not os.path.exists(pth):
            with open(pth, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(file_header)


def generate_csv(which_csv):

    print("Generating MoveNet CSV file for:", which_csv)

    if which_csv == "Training":
        temp = "training"
    else:
        temp = "testing"

    poses = ["standing", "sitting", "lying"]

    for pose_class in range(3):
        image_directory = f"../{temp}/{poses[pose_class]}"

        images = [join(image_directory, f) for f in os.listdir(image_directory) if (isfile(join(image_directory, f)) and f.endswith('.png'))]

        for img_path in images:
            # Reshape image for processing
            img = cv2.imread(img_path) # BGR
            img = img[:,:,::-1] # RGB
                
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
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
                    with open(f"./Public_URFall_{which_csv}.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(pose_row)
            except:
                pass


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
    elif inp == 'B':
        generate_csv("Test")

    print("Done!")
    print('-'*75)
    return 0

if __name__ == '__main__':
    main()
        