import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
image = cv2.imread('face detection/anhcauthu.jpg')
plt.imshow(image[:, :, ::-1])
plt.axis('off')
plt.show()
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
face_coordinates = facecascade.detectMultiScale(gray,1.3,4)
for (a,b,w,h) in face_coordinates:
    cv2.rectangle(image,(a,b),(a+w,b+h),(255,0,0),2)

print("face detection's coordinate: ",face_coordinates)
cv2.imshow('frames',image)
cv2.waitKey(0)