import os
import cv2 as c
import numpy as np
import matplotlib.pyplot as plt
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve
 

class objectDetection():
    URL = r"https://www.dropbox.com/s/xoomeq2ids9551y/opencv_bootcamp_assets_NB13.zip?dl=1"
    asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB13.zip")

    if not os.path.exists(asset_zip_path):
        print(f"Downloading and extracting assests....", end="")

        urlretrieve(URL, asset_zip_path)

        try:
            with ZipFile(asset_zip_path) as z:
                z.extractall(os.path.split(asset_zip_path)[0])
            print("Done")

        except Exception as e:
            print("\nInvalid file.", e)

    classFile  = "coco_class_labels.txt"

    with open(classFile,"r") as fp:
        labels = fp.read().split("\n")

    modelFile  = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
    configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
    net=c.dnn.readNetFromTensorflow(modelFile,configFile)

    def __init__(self):
        pass

    def liveDetection(self):
        thresh=0.5
        net=self.net
        s=0
        if len(sys.argv)>1:
            s=sys.argv[1]
        src=c.VideoCapture(s)
        window="Object Detection Using TF (Live)"
        c.namedWindow(window)
        while c.waitKey(1)!=27:
            bool,frame=src.read()
            frame=c.flip(frame,1)
            if not bool:
                print("Error in Reading Frame")
                continue
            else:
                """ c.imshow(window,c.flip(frame,1)) """
                height=frame.shape[0]
                width=frame.shape[1]
                blob=c.dnn.blobFromImage(frame,1.0,size=(300,300),mean=[0,0,0],swapRB=True,crop=False)
                net.setInput(blob)
                detections=net.forward()
                for i in range(detections.shape[2]):
                    labelId=int(detections[0,0,i,1])
                    confidence=float(detections[0,0,i,2])
                    if confidence > thresh:
                        x_top_left=int(detections[0,0,i,3]*width)
                        y_top_left=int(detections[0,0,i,4]*height)
                        x_bottom_right=int(detections[0,0,i,5]*width)
                        y_bottom_right=int(detections[0,0,i,6]*height)
                        c.putText(frame,f"{self.labels[labelId]}",(x_top_left,(y_top_left-5)),c.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1,c.LINE_AA)
                        c.rectangle(frame,(x_top_left,y_top_left),(x_bottom_right,y_bottom_right),(255,255,255),thickness=2,lineType=c.LINE_AA)
                c.imshow(window,frame)
        c.destroyWindow(window)
        """ print("camera") """
        
    def inputImage(self):
            objSet=set({})
            objList=list([])
            objDict=dict({})
            thresh=0.5
            net=self.net
            """ window="Object Detection Using TF (Input Image)"
            c.namedWindow(window) """
            print("\n**Select pre-loaded Sample Image**")
            im_list=os.listdir("C:\Projects\OpenCV\images")
            print(im_list)
            for i in range(len(im_list)):
                print(f"{i+1}. {im_list[i]}")
            num=int(input("Select Image \n"))
            num=num-1
            image=im_list[num]
            imPath="C:\Projects\OpenCV\images\\"+image
            try:
                im_Arr=c.imread(imPath)
                height=im_Arr.shape[0]
                width=im_Arr.shape[1]
                blob=c.dnn.blobFromImage(im_Arr,1.0,size=(300,300),mean=[0,0,0],swapRB=True,crop=False)
                net.setInput(blob)
                object=net.forward()
                for i in range(object.shape[2]):
                    imId=int(object[0,0,i,1])
                    conf=float(object[0,0,i,2])
                    if conf > thresh:
                        xtop=int(object[0,0,i,3]*width)
                        ytop=int(object[0,0,i,4]*height)
                        xbottom=int(object[0,0,i,5]*width)
                        ybottom=int(object[0,0,i,6]*height)
                        c.putText(im_Arr,f"{self.labels[imId]}",(xtop,ytop-5),c.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1,c.LINE_AA)
                        c.rectangle(im_Arr,(xtop,ytop),(xbottom,ybottom),(255,255,255),thickness=2,)
                        objSet.add(self.labels[imId])
                        objList.append(self.labels[imId])
                c.imshow("Image Extraction",im_Arr)
                print("Object Identified in the Image")
                for x in objSet:
                    objDict[x]=objList.count(x)
                print(objDict)
                c.waitKey(0)
                c.destroyAllWindows()
            except:
                print("Unable to read the input image....")

            
Obj_det=objectDetection()
print("\n ***Object Detection Using OpenCV and TensorFLow with Pre-Trained Model*** \n")
print("1. Detect Live Objects Through Camera \n2. Detect Objects Through input Image")
select=int(input("Select Object Detection Method \n"))
if select==1:
    Obj_det.liveDetection()
elif select==2:
    Obj_det.inputImage()
else:
    print("Select Valid Method")



