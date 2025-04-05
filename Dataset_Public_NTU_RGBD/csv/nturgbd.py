import csv

stand_count = 0
sit_count = 0

with open('NTURGBD_Training.csv', 'r') as in_file:
    reader = csv.reader(in_file)

    header = next(reader)

    with open('NTURGBD_Train.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    with open('NTURGBD_Validation.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for line in reader:
        if int(line[0]) == 0: # Standing
            if stand_count < 14061:
                with open('NTURGBD_Train.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
                    stand_count += 1
            else:
                with open('NTURGBD_Validation.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
                    stand_count += 1
        else: # Sitting
            if sit_count < 16951:
                with open('NTURGBD_Train.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
                    sit_count += 1
            else:
                with open('NTURGBD_Validation.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
                    sit_count += 1