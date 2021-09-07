import cv2
import time
import pathlib
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming


def compute_repeat_LE(des1,des2,kps1,kps2,GT_H_mat_param,w,h,nbr_trd):
    x1_cnt = 0
    x2_cnt = 0
    overlap_cnt = 0
    accu_distance = 0

    GT_H_mat_param =  np.vstack((GT_H_mat_param,np.array((0,0,1))))
    GT_H_mat_param_inv = np.linalg.inv(np.matrix(GT_H_mat_param))
    kp2_new_within = []
    kp1_new_within = []

    for i in range(kps1.shape[0]):
        kp1 = kps1[i]

        kp2_new = np.matmul(GT_H_mat_param,np.hstack((kp1,1)))


        if kp2_new[0]>=0 and kp2_new[0]<=w and kp2_new[1]>=0 and kp2_new[1]<=h: #this can avoid the bordering effect?
            x1_cnt += 1  #proj kp1 on img2
            kp2_new_within.append((i,kp2_new))





    for i in range(kps2.shape[0]):
        kp2 = kps2[i]

        kp1_new = np.matmul(np.array(GT_H_mat_param_inv),np.hstack((kp2,1)))


        if kp1_new[0]>=0 and kp1_new[0]<=w and kp1_new[1]>=0 and kp1_new[1]<=h:
            x2_cnt += 1  #proj kp2 on img1, all of 500 points should still remain within the w*h frame! so the smaller value shoudl be x1_cnt
            kp1_new_within.append((i,kp1_new))


    if x1_cnt<=x2_cnt:
        denorm = x1_cnt
        for p2 in kp2_new_within:
            for j in range(len(kps2)):
                p2_old = kps2[j]
                # print((p2_old-p2[:2])[0],(p2_old-p2[:2])[1])
                if np.abs((p2_old-p2[1][:2])[0])<=nbr_trd and np.abs((p2_old-p2[1][:2])[1])<=nbr_trd:
                    overlap_cnt += 1
                    distance = np.linalg.norm(p2_old-p2[1][:2])
                    accu_distance += distance
                    data_pair = (j,p2_old,p2[0],p2[1][:2])  # maybe add des distance here!
                    break
    else:
        denorm = x2_cnt
        for p1 in kp1_new_within:
            for j in range(len(kps1)):
                p1_old = kps1[j]
                # print((p2_old-p2[:2])[0],(p2_old-p2[:2])[1])
                if np.abs((p1_old-p1[1][:2])[0])<=nbr_trd and np.abs((p1_old-p1[1][:2])[1])<=nbr_trd:
                    overlap_cnt += 1
                    distance = np.linalg.norm(p1_old-p1[1][:2])
                    accu_distance += distance
                    data_pair = (j,p1_old,p1[0],p1[1][:2])  # maybe add des distance here!
                    break

#x1_cnt,x2_cnt,overlap_cnt,repeatibility,avg_loc_distance    # the nbr_trd will effect the LE error.
#  for this part, it doesn't involve any descriptor matching thing
    if denorm>0 and overlap_cnt >0:
        return x1_cnt,x2_cnt,overlap_cnt,float(overlap_cnt)/denorm,float(accu_distance)/overlap_cnt
    else:
        return x1_cnt,x2_cnt,overlap_cnt,None,None


def compute_HE(GT_H_mat_param,esti_H_mat_param,kps1,des1,w,h):
    Homo_error = 0


    GT_H_mat_param = np.vstack((GT_H_mat_param,np.array((0,0,1))))

    for i in range(kps1.shape[0]):
        kp1 = kps1[i]
        kp2_new = np.matmul(GT_H_mat_param,np.hstack((kp1,1)))  # 3*1
        kp2_new_esti = np.matmul(esti_H_mat_param,np.hstack((kp1,1)))
        Homo_error += np.linalg.norm(kp2_new_esti-kp2_new)
    # print("the DELTA H is ",float(Homo_error),kps1.shape[0],"\n",esti_H_mat_param,"\n",GT_H_mat_param)

    Homo_error = float(Homo_error)/kps1.shape[0]

    #  for this part, it doesn't involve any descriptor matching thing
    return Homo_error

def compute_MA(matches,des1,des2,kps1,kps2,GT_H_mat_param,w,h):
    num4matches = matches.shape[0]
    correct_matches_cnt = 0
    for data_row in matches:
        #[match.queryIdx,kp1[match.queryIdx].pt[0],kp1[match.queryIdx].pt[1],match.trainIdx,kp2[match.trainIdx].pt[0],kp2[match.trainIdx].pt[1],match.distance]
        min_des_dis_idx_in_kps2 = data_row[3]
        kp1 = data_row[1:3]
        kp1_proj = np.matmul(GT_H_mat_param,np.hstack((kp1,1)))[:2]
        min_phy_dis = 100000
        min_phy_dis_idx_in_kps2 = 100000
        if kp1_proj[0]>=0 and kp1_proj[0]<=w and kp1_proj[1]>=0 and kp1_proj[1]<=h: #this can avoid the bordering effect?
            for kp2_idx in range(kps2.shape[0]):
                kp2 = kps2[kp2_idx]
                if np.linalg.norm(kp1_proj-kp2)< min_phy_dis:
                    min_phy_dis = np.linalg.norm(kp1_proj-kp2)
                    min_phy_dis_idx_in_kps2 = kp2_idx
            if min_phy_dis_idx_in_kps2 == min_des_dis_idx_in_kps2:
                correct_matches_cnt += 1
    matching_accuracy = float(correct_matches_cnt)/num4matches






#Matching_accuracy,correct_cnt,num4matches
    return matching_accuracy,correct_matches_cnt,num4matches




if __name__=="__main__":


    ori_img_dir = r"E:\Datasets\surgical\final_ori_imgs\\"
    new_img_dir = r"E:\Datasets\surgical\out_imgs\\"

    nbr_trd = 1  #  a higher trd can is not less strict, consequently the repeatabiltiy gets improved.

    ori_img_paths = os.listdir(ori_img_dir)
    transform_list = ["rot","scale","blur","illu"]
    transform_params_list = [ [30,60,90,120,150,180], [0.25,0.5,0.75,1.25,1.5,1.75],[2,4,6,8,10,12], [0.4,0.6,0.8,1.2,1.4,1.6]]
    methods_list = ["SuperPoint","ORB","GFTT_SIFT","AGAST_SIFT"]
    for method in methods_list:
        print("Evaluating all imgs wrt method:  "+method,"\n")


        for img_path in ori_img_paths:
            # all4img = list(Path(new_img_dir+img_path[:-4],f"rot\\").rglob("*.png"))
            img = cv2.imread(ori_img_dir+img_path)
            w = img.shape[1]
            h = img.shape[0]

            for transform,transform_params in zip(transform_list,transform_params_list):
                # transform  = "rot"
                kps_des_path = str(new_img_dir)+img_path[:-4]+"//"+transform+"_pair//"+method+"_kps_des"+".npz"
                Hs_esti_path = str(new_img_dir)+img_path[:-4]+"//"+transform+"_pair//"+method+"_esti_H"+".npz"
                GT_Hs_mat_path = str(new_img_dir)+img_path[:-4]+"//"+transform+"//GT_H_mat.npz"
                Matches_path = str(new_img_dir)+img_path[:-4]+"//"+transform+"_pair//"+method+"_matches"+".npz"

                kps_des_mat = np.load(kps_des_path)#7
                Hs_esti = np.load(Hs_esti_path,allow_pickle=True)
                GT_Hs_mat = np.load(GT_Hs_mat_path)#6 #(Path(new_img_dir,f"GT_H_mat.npz"))#+"\\"+transform+"_pair\\"+
                Matches = np.load(Matches_path)
                kps_des_mat_ori = kps_des_mat["0"]

                for param in transform_params:

                    try:
                        kps_des_mat_param = kps_des_mat[str(param)]# 500 34  idx: x y des
                    except KeyError:
                        print("param: ",param," Ooops,empty!   skip..." )
                        continue
                    GT_H_mat_param = GT_Hs_mat[str(param)]
                    esti_H_mat_param = Hs_esti[str(param)]

                    matches = Matches[str(param)]

                    kps1 = kps_des_mat_ori[:,:2]
                    kps2 = kps_des_mat_param[:,:2]
                    des1 = kps_des_mat_ori[:,2:]
                    des2 = kps_des_mat_param[:,2:]

                    x1_cnt,x2_cnt,overlap_cnt,repeatibility,avg_loc_distance = compute_repeat_LE(des1,des2,kps1,kps2,GT_H_mat_param,w,h,nbr_trd)#   nbr_thrd can be 1 /2/3 pixels

                    # tolerance_trd = 1
                    Homo_error = compute_HE(GT_H_mat_param,esti_H_mat_param,kps1,des1,w,h)
                        # if Homo_error > 100:
                        #     im_out = cv2.warpPerspective(img, esti_H_mat_param,(img.shape[1],img.shape[0])) #new_img
                        #     cv2.imshow("",im_out)
                        #     cv2.waitKey(0)
                        # if Homo_error < 0.1:
                        #     pass

                    Matching_accuracy,correct_cnt,num4matches = compute_MA(matches,des1,des2,kps1,kps2,GT_H_mat_param,w,h)

                    if repeatibility is not None:
                        print("param: ",param," repeat: ",repeatibility," avg_phy_dis: ",avg_loc_distance, " Homo Error: ",Homo_error," matching accu: ",Matching_accuracy, correct_cnt,num4matches)#, "  x1_cnt, x2_cnt, overlap_cnt: ",x1_cnt,x2_cnt,overlap_cnt)
                    else:
                        print("param: ",param," Ooops, denorm=0 or overlap_cnt =0,  skip... ")



                print("method: ",method,"done with "+transform +" for img: ",img_path,"\n")












