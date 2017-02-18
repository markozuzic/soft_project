#Kao argument proslediti file path foldera sa slikama i file path prediktora

import sys
import os
import dlib
import glob
from skimage import io
import cv2
import numpy as np
from sklearn import neighbors
import math
import pandas

def euclideanDistance(instance1, instance2):
    distance = pow((instance1.x - instance2.x), 2) + pow((instance1.y - instance2.y), 2)
    return math.sqrt(distance)

if (len(sys.argv[1:]) == 2):
    arg1, arg2, arg3 = (sys.argv)

faces_folder_path = arg2
predictor_path = arg3

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


dataframe = pandas.read_csv('test2b.csv', header=None)
dataset = dataframe.values
X = dataset[:,0:5].astype(float)
Y = dataset[:,5]
knn = neighbors.KNeighborsClassifier()
knn.fit(X, Y)

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = predictor(img, d)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        v = np.median(img)
        sigma = 0.66
        lower = int(max(30, (1.0 - sigma) * v))
        upper = int(min(80, (1.0 + sigma) * v))

        canny = cv2.Canny(img, lower, upper)
        x1 = shape.part(37).x
        y1 = shape.part(29).y
        x2 = shape.part(40).x
        y2 = shape.part(31).y

        x1l = shape.part(43).x
        x2l = shape.part(46).x

        crop = canny[x1:x2, y1:y2]
        crop_levo = canny[x1l:x2l, y1:y2]
        white = np.count_nonzero(crop == 255)
        white_levo = np.count_nonzero(crop_levo == 255)
        total = crop.size + crop_levo.size
        F5 = float(white + white_levo) / total

        leftEye = shape.part(43)
        rightEye = shape.part(40)
        nose = shape.part(34)
        lip = shape.part(67)
        chin = shape.part(9)

        F1 = euclideanDistance(leftEye, rightEye) / euclideanDistance(leftEye, nose)
        F2 = euclideanDistance(leftEye, rightEye) / euclideanDistance(leftEye, lip)
        F3 = euclideanDistance(leftEye, nose) / euclideanDistance(leftEye, chin)
        F4 = euclideanDistance(leftEye, nose) / euclideanDistance(leftEye, lip)

        print knn.predict([[F1, F2, F3, F4, F5]])

    dlib.hit_enter_to_continue()
    print '\n'