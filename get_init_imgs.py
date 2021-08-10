import cv2
import time
import pathlib
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import cv2
import os

#
# path4videos = 'D:\Study\Datasets\cholec80\\videos\\'
# path4current = 'D:\Study\Datasets\cholec80\current\\'
# path4next = 'D:\Study\Datasets\cholec80\\next\\'
path4videos = 'E:\Datasets\\a_LAB\irreg\\'
path4current = 'E:\Datasets\surgical\ori_imgs\\'
# path4next = 'D:\Study\Datasets\hamlyn\\next\\'
videosPathList = []

def getdata(path,name):
    reader = cv2.VideoCapture(path)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    have_more_frame = True
    # domain = [250, 2500, 5000, 10000, 20000, 30000, 40000]  #for cholec80
    domain = [50*2, 100*2, 150*2, 200*2, 250*2, 300*2]
    k = 0
    frameIdx = 0
    while have_more_frame:
        have_more_frame, frame = reader.read()
        frameIdx += 1
        if frameIdx in domain:
            k += 1
            cv2.imshow('', frame)
            cv2.waitKey(1)
            cv2.imwrite(path4current + name[:-4] +'_'+str(k)+ '.png', frame)
        if k == 6:
            break

    reader.release()
    cv2.destroyAllWindows()




if __name__=="__main__":
    for f_name in os.listdir(path4videos):
        if  f_name.endswith('.mp4') or f_name.endswith('.avi') :
            videosPathList.append(f_name)

    for name in videosPathList:
        path = path4videos + name
        getdata(path,name)

