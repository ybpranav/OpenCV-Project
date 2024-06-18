import cv2 as c
import matplotlib.pyplot as plt
import sys
import os
import urllib
s=0
if len(sys.argv)>1:
    s=sys.argv[1]
if not os.path.isfile('goturn.prototext') or not os.path.isfile('goturn.caffemodel'):
    print("Downloading goturn model")
    urllib.request.urlret 
def drawRectangle(frame,box):
    framecopy=c.rectangle(frame,(int(box[0]),int(box[1])),(int(box[0])+int(box[2]),int(box[1])+int(box[3])),(0,255,0),thickness=3,lineType=c.LINE_AA)
    return framecopy
src=c.VideoCapture(s)
bool,frame=src.read()
""" c.imshow("Object Detection",frame) """
while c.waitKey(1)!=27:
    box=c.selectROI(frame,False)
    if box:
        break
    elif not box:
        print("Unable to select box area")
c.destroyWindow("ROI selector")
print("The box area Selected is",box)

framecopy=drawRectangle(frame,box)
""" tracker=c.TrackerKCF_create() """
tracker=c.TrackerKCF.create()
ok=tracker.init(frame,box)
c.namedWindow("Detection 2")
while c.waitKey(1)!=27:
    ok,frame=src.read()
    if not ok:
        print("Cannot read Frame")
        break
    else:
        ok,bbox=tracker.update(frame)
        if not ok:
            print("Cannot update frame")
            break
        else:
            frame=drawRectangle(frame,bbox)
            c.imshow("Detection 2",frame)
c.destroyWindow("Detection 2")