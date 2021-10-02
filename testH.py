import cv2
import numpy as np
from math import cos,sin

path = "E:\Datasets\surgical\ori_imgs\\4_5.png"
img = cv2.imread(path)
rows,cols,ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200],[50,20]])
pts2 = np.float32([[10,100],[200,50],[100,250],[50,80]])
# M1 = cv2.getAffineTransform(pts1,pts2)  # 3 pairs
M2 = cv2.findHomography(pts1,pts2)  # 6 pairs
print(M2)
M3 = cv2.getPerspectiveTransform(pts1,pts2)  # 4 pairs
print(M3)
M1 = np.array([1,0,0,
               0,1,0,
               -0.0003,-0.000,1],dtype=np.float64).reshape((3,3))
# print(M1)
dst = cv2.warpPerspective(img,M1,(cols,rows))  # Only receive 2*3 affine transformation
# #
dst = np.concatenate((img, dst), axis=1)
cv2.imshow("",dst)
cv2.waitKey(0)

kp2_new = np.matmul(M1,np.hstack((3,4,1)))
print("kkkk",kp2_new)
kp2_new = (kp2_new/kp2_new[2])[:2]
print("lkl;k;l",kp2_new)

