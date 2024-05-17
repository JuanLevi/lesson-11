import cv2, numpy, os

configfile="cars/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
model="cars/frozen_inference_graph.pb"
labels="cars/yolo3.txt"

file=open(labels,'rt')

objectlabels=file.read()

objectnames=[]

with open(labels,'rt') as file:
    objectnames=file.read().rstrip('\n').split('\n')
    


print(objectnames)



"""
count=1

while count>0:
    ret, frame=cam.read()
    
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)



    face=facexml.detectMultiScale(grey,3,6)
    print(face)

    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)


    eye=eyexml.detectMultiScale(grey,3,6)
    print(eye)

    for (x,y,w,h) in eye:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)



    cv2.imshow("webcam",frame)
    k=cv2.waitKey(10)
    if k==27:
        break

"""