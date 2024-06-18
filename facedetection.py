import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve
def download_and_unzip(url, save_path):
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


URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")

# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
    
s=0
if len(sys.argv)>1:
    s=sys.argv[1]
net=cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
cv2.namedWindow("fd")
src=cv2.VideoCapture(s)
twidth=300
theight=300
mean=[104,117,123]
thresh=0.7
while cv2.waitKey(1)!=27:
    bool,frame=src.read()
    frame=cv2.flip(frame,1)
    fheight=frame.shape[0]
    fwidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame,1.0,(twidth,theight),mean,swapRB=False,crop=False)
    net.setInput(blob)
    det=net.forward()
    for i in range(det.shape[2]):
        conf=det[0,0,i,2]
        if conf > thresh:
            xtop=int(det[0,0,i,3]*fwidth)
            ytop=int(det[0,0,i,4]*fheight)
            xbottom=int(det[0,0,i,5]*fwidth)
            ybottom=int(det[0,0,i,6]*fheight)
            cv2.rectangle(frame,(xtop,ytop),(xbottom,ybottom),(0,255,0),lineType=cv2.LINE_AA,thickness=1)
            frame_Crop=frame[ytop:ybottom,xtop:ybottom]
            cv2.imwrite(f"cropped{i}img.jpg",frame_Crop)
    cv2.imshow("fd",frame)
cv2.destroyWindow("fd")