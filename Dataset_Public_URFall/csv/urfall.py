import csv

stand_count = 0
sit_count = 0
lying_count = 0

with open('Public_URFall_Training.csv', 'r') as in_file:
    reader = csv.reader(in_file)

    header = next(reader)

    with open('Public_URFall_Train.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    with open('Public_URFall_Validation.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for line in reader:
        if int(line[0]) == 0: # Standing
            if stand_count < 997:
                with open('Public_URFall_Train.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
                    stand_count += 1
            else:
                with open('Public_URFall_Validation.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
                    stand_count += 1
        elif int(line[0]) == 1: # Sitting
            if sit_count < 1000:
                with open('Public_URFall_Train.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
                    sit_count += 1
            else:
                with open('Public_URFall_Validation.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
                    sit_count += 1
        else: # Lying
            if lying_count < 984:
                with open('Public_URFall_Train.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
                    lying_count += 1
            else:
                with open('Public_URFall_Validation.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(line)
                    lying_count += 1


    