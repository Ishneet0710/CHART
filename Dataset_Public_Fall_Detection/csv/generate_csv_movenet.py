import cv2
import numpy as np
import csv
import os
import tensorflow as tf

# Load MoveNet Model
interpreter = tf.lite.Interpreter(model_path="../../lite-model_movenet_singlepose_lightning_3.tflite")
interpreter.allocate_tensors()

# Datasets used for training, validation and test
trg_dsets = [489, 569, 581, 722, 731, 758, 807, 1219, 1260, 1301, 1373, 1378, 1392, 1790, 1843, 1954]
val_dsets = [1176, 2123]
tst_dsets = [786, 832, 925]

file_header = []
file_header.insert(0, 'class')
for i in range(1, 18):
    file_header.insert(3*i - 2, f'y{i}')
    file_header.insert(3*i - 1, f'x{i}')
    file_header.insert(3*i,     f'c{i}')


def make_csv_files():
    paths = ["./Public_Fall_Detection_Train.csv", "./Public_Fall_Detection_Validation.csv", "./Public_Fall_Detection_Test.csv"]
    for pth in paths:
        if not os.path.exists(pth):
            with open(pth, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(file_header)


def generate_csv(which_csv, dsets_used):

    print("Generating MoveNet CSV file for:", which_csv)

    for dataset in dsets_used:
        print("Current dataset:", dataset)

        # Get labels
        with open(f"../images/Dataset {dataset}/Labels/labels.csv", "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            labels = [row for row in reader]


        for temp1, temp2 in labels:
            img = int(temp1)
            pose_class = int(temp2)
            if pose_class not in [1, 2, 3]: # Not standing, sitting or lying
                continue
            img_path = f"../images/Dataset {dataset}/Images/rgb_{img:04d}.png"

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
                    pose_row.insert(0, pose_class - 1)

                    # Write into csv file
                    with open(f"./Public_Fall_Detection_{which_csv}.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(pose_row)
            except:
                pass


def main():
    print('-'*75)
    print("MoveNet CSV File Generator")
    print("Options:\n\tA: Training Set\n\tB: Validation Set\n\tC: Test Set")
    print('-'*75)
    print("Enter choice: ")

    while True:
        inp = input()
        if inp not in ['A', 'B', 'C']:
            print("Invalid Input")
            continue
        break

    make_csv_files()
    if inp == 'A':
        generate_csv("Train", trg_dsets)
    elif inp == 'B':
        generate_csv("Validation", val_dsets)
    else:
        generate_csv("Test", tst_dsets)

    print("Done!")
    print('-'*75)
    return 0

if __name__ == '__main__':
    main()
        