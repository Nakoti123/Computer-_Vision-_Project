import cv2
import numpy as np
import face_recognition
import os
import dlib

path = 'images'
image = []
classNames = []
myList = os.listdir(path)
#print(myList)
for cl in myList:
     curImg = cv2.imread(f'{path}/{cl}')
     image.append(curImg)
     classNames.append(os.path.splitext(cl)[0])
print(classNames)
def findEncodings(image):
   encodeList = []
   for img in image:
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       encode = face_recognition.face_encodings(img)[0]
       encodeList.append(encode)
   return encodeList
print('Encoding Complete')
encodeListKnown = findEncodings(image)

cap = cv2.VideoCapture(0)
wCam, hCam = 720, 720
frameR = 100  # Frame Reduction
smoothening = 7
while True:
    success, img = cap.read(1)
    imgS = cv2.resize(img,(0,0),None,fx=0.25,fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)

        print(faceDis)
        if matches[matchIndex]:
              name = classNames[matchIndex].upper()
        else: name='unknown'
        print(name)
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img,name,(x1+6,y2-6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
