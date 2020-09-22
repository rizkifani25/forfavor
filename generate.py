import numpy as np
import cv2
import os

from os import path

if not path.exists("images"):
    os.mkdir("images")

uid = input("Input your ID number : ")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

counter = 1

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        dirname = "images/" + str(uid)
        if not path.exists(dirname):
            os.mkdir(dirname)
        img_item = os.path.join(dirname, str(counter) + ".png")
        cv2.imwrite(img_item, roi_color)
        counter = counter + 1
    if counter > 20:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()