#Kao argument proslediti file path shape predictor-a

import dlib
import cv2
import numpy as np
from sklearn import neighbors
import math
import pandas
import sys

def euclideanDistance(instance1, instance2):
    distance = pow((instance1.x - instance2.x), 2) + pow((instance1.y - instance2.y), 2)
    return math.sqrt(distance)

if (len(sys.argv[1:]) == 1):
    arg1, arg2 = (sys.argv)

predictor_path = arg2
dataframe = pandas.read_csv("test2b.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:5].astype(float)
Y = dataset[:,5]
knn = neighbors.KNeighborsClassifier()
knn.fit(X, Y)

vc = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

while(True):
    ret, frame = vc.read()
    dets = detector(frame, 0)
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(frame, d)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        v = np.median(frame)
        sigma = 0.66
        lower = int(max(30, (1.0 - sigma) * v))
        upper = int(min(80, (1.0 + sigma) * v))

        canny = cv2.Canny(frame, lower, upper)
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
        total = crop.size + crop_levo.size + 1
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

        if (str(knn.predict([[F1, F2, F3, F4, F5]])) == '[\'children\']'):
            age_group = 'child'
        elif(str(knn.predict([[F1, F2, F3, F4, F5]])) == '[\'adults\']'):
            age_group = 'adult'
        else:
            age_group = 'senior'

        cv2.putText(frame, age_group, ((d.left()) + 10, (d.top()) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2)


    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27:
        break

vc.release()
cv2.destroyAllWindows()
