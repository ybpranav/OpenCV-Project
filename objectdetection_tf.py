import os
import cv2 as c
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve
""" def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)
URL = r"https://www.dropbox.com/s/xoomeq2ids9551y/opencv_bootcamp_assets_NB13.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB13.zip")

# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path) """
classFile  = "C:\Projects\OpenCV\coco_class_labels.txt"
with open(classFile,"r") as fp:
    labels = fp.read().split("\n")
""" print(labels) """
""" modelFile  = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt") """
modelFile="C:\Projects\OpenCV\models\ssd_mobilenet_v2_coco_2018_03_29\frozen_inference_graph.pb"
configFile="C:\Projects\OpenCV\models\ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
net=c.dnn.readNetFromTensorflow(modelFile,configFile)
class objectDetection():
    def __init__(self,net,im):
        self.net=net
        self.img=im
Obj_det=objectDetection(net)
print("***Object Detection Using OpenCV and TensorFLow with Pre-Trained Model*** \n")
select=int(input("Select Object Detection Method"))


