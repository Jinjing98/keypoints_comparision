import cv2
import time
import pathlib
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def create_rot_new(ori_img_paths,ori_img_dir,new_img_dir_general):
    GT_H_mat = {}
    for img_path in ori_img_paths:
        img = cv2.imread(ori_img_dir+img_path)
        # new_img_dir = new_img_dir+"/"+img_path[:-3]+"/rot/"
        new_img_dir = Path(new_img_dir_general+"/"+img_path[:-3], f"rot/")
        new_img_dir.mkdir(parents = True, exist_ok = True)
        new_img_dir_pair = Path(new_img_dir_general+"/"+img_path[:-3], f"rot_pair/")
        new_img_dir_pair.mkdir(parents = True, exist_ok = True)

        y,x = img.shape[:2]
        for ang in [30,60,90,120,150,180]:
            rotate = cv2.getRotationMatrix2D((x/2,y/2),ang,1)  # the 3rd row is  0 0 1
            #  it is the same as get affine transform?
            img_rot = cv2.warpAffine(img,rotate,(x,y))
            cv2.imwrite(str(new_img_dir)+"\\"+str(ang)+".png", img_rot)
            GT_H_mat[str(ang)] = rotate
        GT_H_mat_path = Path(new_img_dir,f"GT_H_mat.npz")
        np.savez(GT_H_mat_path,**GT_H_mat)

def create_scale_new(ori_img_paths,ori_img_dir,new_img_dir_general):
    GT_H_mat = {}
    for img_path in ori_img_paths:
        img = cv2.imread(ori_img_dir+img_path)
        # new_img_dir = new_img_dir+"/"+img_path[:-3]+"/scale/"
        new_img_dir = Path(new_img_dir_general+"/"+img_path[:-3], f"scale/")
        new_img_dir.mkdir(parents = True, exist_ok = True)
        new_img_dir_pair = Path(new_img_dir_general+"/"+img_path[:-3], f"scale_pair/")
        new_img_dir_pair.mkdir(parents = True, exist_ok = True)

        y,x = img.shape[:2]
        for scale in [0.25,0.5,0.75,1.25,1.5,1.75]:
            rotate = cv2.getRotationMatrix2D((x/2,y/2),0,scale)  # the 3rd row is  0 0 1
            img_scale = cv2.warpAffine(img,rotate,(x,y))
            cv2.imwrite(str(new_img_dir)+"\\"+str(scale)+".png", img_scale)
            GT_H_mat[str(scale)] = rotate
        GT_H_mat_path = Path(new_img_dir,f"GT_H_mat.npz")
        np.savez(GT_H_mat_path,**GT_H_mat)


def create_illu_new(ori_img_paths,ori_img_dir,new_img_dir_general):# gamma
    GT_H_mat = {}
    for img_path in ori_img_paths:
        img = cv2.imread(ori_img_dir+img_path)
        # new_img_dir = new_img_dir+"/"+img_path[:-3]+"/illu/"
        new_img_dir = Path(new_img_dir_general+"/"+img_path[:-3], f"illu/")
        new_img_dir.mkdir(parents = True, exist_ok = True)
        new_img_dir_pair = Path(new_img_dir_general+"/"+img_path[:-3], f"illu_pair/")
        new_img_dir_pair.mkdir(parents = True, exist_ok = True)

        y,x = img.shape[:2]

        for gamma in [0.4,0.6,0.8,1.2,1.4,1.6]:
            invGamma = 1.0 /gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
            img_illu = cv2.LUT(img, table)
            cv2.imwrite(str(new_img_dir)+"\\"+str(gamma)+".png", img_illu)
            rotate = cv2.getRotationMatrix2D((x/2,y/2),0,1)
            GT_H_mat[str(gamma)] = rotate
        GT_H_mat_path = Path(new_img_dir,f"GT_H_mat.npz")
        np.savez(GT_H_mat_path,**GT_H_mat)


def create_blur_new(ori_img_paths,ori_img_dir,new_img_dir_general):
    GT_H_mat = {}
    for img_path in ori_img_paths:
        img = cv2.imread(ori_img_dir+img_path)
        # new_img_dir = new_img_dir+"/"+img_path[:-3]+"/blur/"
        new_img_dir = Path(new_img_dir_general+"/"+img_path[:-3], f"blur/")
        new_img_dir.mkdir(parents = True, exist_ok = True)
        new_img_dir_pair = Path(new_img_dir_general+"/"+img_path[:-3], f"blur_pair/")
        new_img_dir_pair.mkdir(parents = True, exist_ok = True)

        y,x = img.shape[:2]
        for size in [(2,2),(4,4),(6,6),(8,8),(10,10),(12,12)]:
            img_blur = cv2.blur(img,size)
            cv2.imwrite(str(new_img_dir)+"\\"+str(size[0])+".png", img_blur)
            rotate = cv2.getRotationMatrix2D((x/2,y/2),0,1)
            GT_H_mat[str(size[0])] = rotate
        GT_H_mat_path = Path(new_img_dir,f"GT_H_mat.npz")
        np.savez(GT_H_mat_path,**GT_H_mat)

def get_kp_des_match(transform,transform_params,kp1_des1_np,kp1,des1):
        kps_des_mat = {}
        matches_mat = {}
        H_mat = {}
        kps_des_mat[str(0)] = kp1_des1_np
        for new_img_param in transform_params:
            new_img_name = str(new_img_dir)+img_path[:-4]+"\\"+transform+"\\"+str(new_img_param)+".png"
            new_img = cv2.imread(str(new_img_name))
            kp2, des2 = orb.detectAndCompute(new_img,None)
            kp2_des2_np = np.zeros((len(kp2),34))
            id = 0
            for kp in kp2:
                kp2_des2_np[id] = [kp.pt[0],kp.pt[1]]+list(des2[id])
                id += 1

            kps_des_mat[str(new_img_param)] = kp2_des2_np


            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)
            num4matches = len(matches)
            matches = matches[:num4matches]
            # print(matches[0].queryIdx)

            idx = 0
            matches_np = np.empty((num4matches,7))
            for match in matches:
                # matches_np[idx] = [match.queryIdx,match.trainIdx,match.distance]
                #kp1[m.queryIdx].pt

                matches_np[idx] = [match.queryIdx,kp1[match.queryIdx].pt[0],kp1[match.queryIdx].pt[1],match.trainIdx,kp2[match.trainIdx].pt[0],kp2[match.trainIdx].pt[1],match.distance]

                idx += 1
            matches_mat[str(new_img_param)] = matches_np

            if len(matches) <20:
                print("Warning! "+str(len(matches))+" is less than 20 good matches!"+"\n the path is : "+ new_img_name)
            # if new_img_name == "E:\Datasets\surgical\out_imgs\\18_3\\blur\\12.png":
            #     print("")
            img3 = cv2.drawMatches(img,kp1,new_img,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img3 = cv2.resize(img3,(1280,360))#720 1280
            matched_img_path = str(new_img_dir)+img_path[:-4]+"\\"+transform+"_pair\\"+str(new_img_param)+"_ORB"+".png"
            cv2.imwrite(matched_img_path,img3)

            src_pts = matches_np[:,1:3]
            dst_pts = matches_np[:,4:6]

            H,status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            # im_out = cv2.warpPerspective(img, H,(img.shape[1],img.shape[0])) #new_img
            H_mat[str(new_img_param)] = H



            # if transform == "rot":
            #     src_pts = matches_np[:,1:3]
            #     dst_pts = matches_np[:,4:6]
            #
            #     H,status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            #     # im_out = cv2.warpPerspective(img, H,(img.shape[1],img.shape[0])) #new_img
            #     H_mat[str(new_img_param)] = H
            #     # GT_H_path = str(new_img_dir)+img_path[:-4]+"\\"+transform+"\\GT_H_mat.npz"
            #     # GT_H_mat = np.load(GT_H_path)
            #     #  src_pts_xy dst_pts_xy  gt_H
            # if transform == "scale":
            #     src_pts = matches_np[:,1:3]
            #     dst_pts = matches_np[:,4:6]
            #
            #     H,status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            #     # im_out = cv2.warpPerspective(img, H,(img.shape[1],img.shape[0])) #new_img
            #     H_mat[str(new_img_param)] = H




        des_mat_path = str(new_img_dir)+img_path[:-4]+"\\"+transform+"_pair\\"+"ORB_kps_des"+".npz"
        matches_mat_path = str(new_img_dir)+img_path[:-4]+"\\"+transform+"_pair\\"+"ORB_matches"+".npz"
        np.savez(des_mat_path,**kps_des_mat)
        np.savez(matches_mat_path,**matches_mat)
        H_mat_path = str(new_img_dir)+img_path[:-4]+"\\"+transform+"_pair\\"+"ORB_esti_H"+".npz"
        np.savez(H_mat_path,**H_mat)




if __name__=="__main__":

    ori_img_dir = r"E:\Datasets\NCT\ori_imgs\\"
    new_img_dir = r"E:\Datasets\NCT\out_imgs\\"
    #
    ori_img_dir = r"E:\Datasets\surgical\ori_imgs\\"
    new_img_dir = r"E:\Datasets\surgical\out_imgs\\"






    ori_img_paths = os.listdir(ori_img_dir)
    create_rot_new(ori_img_paths,ori_img_dir,new_img_dir)
    create_scale_new(ori_img_paths,ori_img_dir,new_img_dir)
    create_illu_new(ori_img_paths,ori_img_dir,new_img_dir)
    create_blur_new(ori_img_paths,ori_img_dir,new_img_dir)

    print("data collection done.\n start processing ...")


    for img_path in ori_img_paths:
        # all4img = list(Path(new_img_dir+img_path[:-4],f"rot\\").rglob("*.png"))
        img = cv2.imread(ori_img_dir+img_path)
        orb = cv2.ORB_create(nfeatures=500)

        kp1, des1 = orb.detectAndCompute(img,None)  # maybe less than 500
        num4kps = len(kp1)
        kp1_des1_np = np.empty((num4kps,34))
        id = 0
        for kp in kp1:
            kp1_des1_np[id] = [kp.pt[0],kp.pt[1]]+list(des1[id])
            id += 1
        kps_des_mat = {}
        matches_mat = {}
        kps_des_mat[str(0)] = kp1_des1_np


        method_list = ["ORB","KP2D"]
        transform_list = ["rot","scale","blur","illu"]
        transform_params_list = [ [30,60,90,120,150,180], [0.25,0.5,0.75,1.25,1.5,1.75],[2,4,6,8,10,12], [0.4,0.6,0.8,1.2,1.4,1.6]]
        method = "ORB"
        for transform,transform_params in zip(transform_list,transform_params_list):
            get_kp_des_match(transform,transform_params,kp1_des1_np,kp1,des1)





        #
        #
        #
        #
        # for new_img_param in [30,60,90,120,150,180]:
        #     new_img_name = str(new_img_dir)+img_path[:-4]+"\\rot\\"+str(new_img_param)+".png"
        #     new_img = cv2.imread(str(new_img_name))
        #     kp2, des2 = orb.detectAndCompute(new_img,None)
        #     kp2_des2_np = np.zeros_like(kp1_des1_np)
        #     id = 0
        #     for kp in kp2:
        #         kp2_des2_np[id] = list(kp.pt).append(des2[id])
        #         id += 1
        #     kps_des_mat[str(new_img_param)] = kp2_des2_np
        #
        #
        #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #     matches = bf.match(des1,des2)
        #     matches = sorted(matches, key = lambda x:x.distance)
        #     num4matches = 100
        #     matches = matches[:num4matches]
        #     # print(matches[0].queryIdx)
        #
        #     idx = 0
        #     matches_np = np.empty((num4matches,3))
        #     for match in matches:
        #         matches_np[idx] = [match.queryIdx,match.trainIdx,match.distance]
        #         idx += 1
        #     matches_mat[str(new_img_param)] = matches_np
        #
        #     if len(matches) <100:
        #         print("Warning! less than 100 good matches!")
        #     img3 = cv2.drawMatches(img,kp1,new_img,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #     img3 = cv2.resize(img3,(1280,360))#720 1280
        #     matched_img_path = str(new_img_dir)+img_path[:-4]+"\\rot_pair\\"+str(new_img_param)+"_ORB"+".png"
        #     cv2.imwrite(matched_img_path,img3)
        #
        #
        # des_mat_path = str(new_img_dir)+img_path[:-4]+"\\rot_pair\\"+"ORB_kps_des"+".npz"
        # matches_mat_path = str(new_img_dir)+img_path[:-4]+"\\rot_pair\\"+"ORB_matches"+".npz"
        # np.savez(des_mat_path,**kps_des_mat)
        # np.savez(matches_mat_path,**matches_mat)













