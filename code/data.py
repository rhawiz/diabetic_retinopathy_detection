import cv2
import csv
import os
import numpy as np


def load_data(folder_paths, labels_path, log=False, limit=-1):
    X = []
    y = []
    labels = {}
    with open(labels_path, 'rb') as csvfile:
        for row in csv.reader(csvfile):
            labels[row[0]] = row[1]
    count = 0
    for folder_path in folder_paths:
        for subdir, dirs, files in os.walk(folder_path):
            for file in files:
                count += 1
                path = os.path.join(subdir, file)
                if '.' not in file:
                    continue

                label, file_type = file.split('.')

                if log:
                    print "---File %s" % file
                image = cv2.imread(path)

                if log:
                    print "\t--Resizing..."
                image_resized = cv2.resize(image, (300, 300))

                try:
                    y.append(labels[label])
                    X.append(image_resized)
                except KeyError:
                    continue
                # TODO: Normalization
                if count == limit:
                    break

    if log:
        print "-Converting"

    # Theano works with fp32 precision
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int32)

    if log:
        print "-Reshaping"

    X = X.reshape(
        -1,  # number of samples, -1 makes it so that this number is determined automatically
        3,  # 3 color channel, since images are only black and white
        300,  # first image dimension (vertical)
        300,  # second image dimension (horizontal)
    )

    return X, y