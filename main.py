import cv2, numpy, os

configfile="cars/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
model="cars/frozen_inference_graph.pb"
labels="cars/yolo3.txt"
frame1="cars/frame.jpg"
frame2="cars/animals.jpg"
frame3="cars/dinner.jpg"

file=open(labels,'rt')

objectlabels=file.read()

objectnames=[]

with open(labels,'rt') as file:
    objectnames=file.read().rstrip('\n').split('\n')
    
print(objectnames)

frame=cv2.imread(frame3)
rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


detection=cv2.dnn_DetectionModel(model,configfile) 

#model training

detection.setInputSize(350,350)
detection.setInputScale(1.0/127.5)
detection.setInputMean((127.5,127.5,127.5))
detection.setInputSwapRB(True)

print(detection.detect(rgb,confThreshold=0.5))

classIndex,confidence,boxDetails = detection.detect(rgb,confThreshold=0.5)


for classi,conf,box in zip(classIndex.flatten(),confidence.flatten(),boxDetails):
    cv2.rectangle(frame,box,(0,0,255),4)
    cv2.putText(frame,objectnames[classi-1],(box[0]+10,box[1]+40),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),4)


cv2.imshow("Image",frame)
cv2.waitKey(0)


