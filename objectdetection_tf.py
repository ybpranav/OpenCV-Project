import os
import cv2 as c
import numpy as np
import matplotlib.pyplot as plt
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

    def cam(self):
        print("camera")
        
    def im(self):
        print("Image")

Obj_det=objectDetection()
print("\n ***Object Detection Using OpenCV and TensorFLow with Pre-Trained Model*** \n")
print("1. Detect Live Objects Through Camera \n2. Detect Objects Through input Image")
select=int(input("Select Object Detection Method \n"))
if select==1:
    Obj_det.cam()
elif select==2:
    Obj_det.im()
else:
    print("Select Valid Method")



