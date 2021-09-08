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

    # GT_H_mat_param =  np.vstack((GT_H_mat_param,np.array((0,0,1))))
    GT_H_mat_param_inv = np.linalg.inv(np.matrix(GT_H_mat_param))
    kp2_new_within = []
    kp1_new_within = []

    for i in range(kps1.shape[0]):
        kp1 = kps1[i]

        kp2_new = np.matmul(GT_H_mat_param,np.hstack((kp1,1)))
        kp2_new = kp2_new/kp2_new[2]


        if kp2_new[0]>=0 and kp2_new[0]<=w and kp2_new[1]>=0 and kp2_new[1]<=h: #this can avoid the bordering effect?
            x1_cnt += 1  #proj kp1 on img2
            kp2_new_within.append((i,kp2_new))





    for i in range(kps2.shape[0]):
        kp2 = kps2[i]

        kp1_new = np.matmul(np.array(GT_H_mat_param_inv),np.hstack((kp2,1)))
        kp1_new = kp1_new/kp1_new[2]



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


    # GT_H_mat_param = np.vstack((GT_H_mat_param,np.array((0,0,1))))

    for i in range(kps1.shape[0]):
        kp1 = kps1[i]
        kp2_new = np.matmul(GT_H_mat_param,np.hstack((kp1,1)))  # 3*1
        kp2_new = kp2_new/kp2_new[2]
        kp2_new_esti = np.matmul(esti_H_mat_param,np.hstack((kp1,1)))
        kp2_new_esti = kp2_new_esti/kp2_new_esti[2]

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
        kp1_proj = np.matmul(GT_H_mat_param,np.hstack((kp1,1)))
        kp1_proj = (kp1_proj/kp1_proj[2])[:2]


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





def get_metrics():

    for method in methods_list:
        print("\n now start to Evaluating all imgs wrt method:  "+method,"...")


        repeatibility_dict = {"rot":[],"scale":[],"illu":[],"blur":[],"proj":[],"mix":[]}  # 4 elements wrt to 4 transforms given certrain kps method, each element is a list with length of total imgs
        avg_loc_distance_dict = {"rot":[],"scale":[],"illu":[],"blur":[],"proj":[],"mix":[]}
        Homo_error_dict = {"rot":[],"scale":[],"illu":[],"blur":[],"proj":[],"mix":[]}
        Matching_accuracy_dict = {"rot":[],"scale":[],"illu":[],"blur":[],"proj":[],"mix":[]}



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


                tem_repeatibility_list = []
                tem_avg_loc_distance_list = []
                tem_Homo_error_list = []
                tem_Matching_accuracy_list = []

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
                    # print("now the para is :",param)
                    x1_cnt,x2_cnt,overlap_cnt,repeatibility,avg_loc_distance = compute_repeat_LE(des1,des2,kps1,kps2,GT_H_mat_param,w,h,nbr_trd)#   nbr_thrd can be 1 /2/3 pixels
                    Homo_error = compute_HE(GT_H_mat_param,esti_H_mat_param,kps1,des1,w,h)
                    Matching_accuracy,correct_cnt,num4matches = compute_MA(matches,des1,des2,kps1,kps2,GT_H_mat_param,w,h)



                    tem_repeatibility_list.append(repeatibility)
                    tem_avg_loc_distance_list.append(avg_loc_distance)
                    tem_Homo_error_list.append(Homo_error)
                    tem_Matching_accuracy_list.append(Matching_accuracy)


                    if repeatibility is not None:
                        print("param: ",param," repeat: ",repeatibility," avg_phy_dis: ",avg_loc_distance, " Homo Error: ",Homo_error," matching accu: ",Matching_accuracy, correct_cnt,num4matches)#, "  x1_cnt, x2_cnt, overlap_cnt: ",x1_cnt,x2_cnt,overlap_cnt)
                    else:
                        print("param: ",param," Ooops, denorm=0 or overlap_cnt =0,  skip... ")
                #certain method: for certain img, certain type of transformation we get one set of avg metrics
                repeatibility_avg = np.nanmean(np.array(tem_repeatibility_list,dtype=np.float64))
                avg_loc_distance_avg = np.nanmean(np.array(tem_avg_loc_distance_list,dtype=np.float64))
                Homo_error_avg = np.nanmean(np.array(tem_Homo_error_list,dtype=np.float64))
                Matching_accuracy_avg = np.nanmean(np.array(tem_Matching_accuracy_list,dtype=np.float64))

                repeatibility_dict[transform].append(repeatibility_avg)
                avg_loc_distance_dict[transform].append(avg_loc_distance_avg)
                Homo_error_dict[transform].append(Homo_error_avg)
                Matching_accuracy_dict[transform].append(Matching_accuracy_avg)

                print("method: ",method,"done with "+transform +" for img: ",img_path,"\n")

        #performance of certain kps method

        print("for method : ",method, " display the metrics for different transformation:  reap LE HE MA \n")
        for transform in transform_list:
            avg_rep = np.nanmean(np.array(repeatibility_dict[transform],dtype=np.float64))
            avg_LE = np.nanmean(np.array(avg_loc_distance_dict[transform],dtype=np.float64))
            avg_HE = np.nanmean(np.array(Homo_error_dict[transform],dtype=np.float64))
            avg_MA = np.nanmean(np.array(Matching_accuracy_dict[transform],dtype=np.float64))
            print(transform,avg_rep,avg_LE,avg_HE,avg_MA)

            all_repeatibility_dict[transform].append(avg_rep)
            all_avg_loc_distance_dict[transform].append(avg_LE)
            all_Homo_error_dict[transform].append(avg_HE)
            all_Matching_accuracy_dict[transform].append(avg_MA)



        # in the order of methods_list for each sub list in the dict
        np.savez(new_img_dir+"all_repeatibility_dict_trd"+str(nbr_trd)+".npz",**all_repeatibility_dict)
        np.savez(new_img_dir+"all_avg_loc_distance_dict_trd"+str(nbr_trd)+".npz",**all_avg_loc_distance_dict)
        np.savez(new_img_dir+"all_Homo_error_dict_trd"+str(nbr_trd)+".npz",**all_Homo_error_dict)
        np.savez(new_img_dir+"all_Matching_accuracy_dict_trd"+str(nbr_trd)+".npz",**all_Matching_accuracy_dict)






if __name__=="__main__":


    ori_img_dir = r"E:\Datasets\surgical\final_ori_imgs\\"
    new_img_dir = r"E:\Datasets\surgical\out_imgs\\"

    # nbr_trd = 5 #[1,2,3,4,5,6] #pixels#  a higher trd can is not less strict, consequently the repeatabiltiy gets improved.
    nbr_trd_list = [1,2,3,4,5]
    ori_img_paths = os.listdir(ori_img_dir)
    transform_list = ["rot","scale","blur","illu","proj","mix"]
    # transform_params_list = [ [30,60,90,120,150,180], [0.25,0.5,0.75,1.25,1.5,1.75],[2,4,6,8,10,12], [0.4,0.6,0.8,1.2,1.4,1.6]]

    rot_list = [10,15,20,80,85,90]
    scale_list = [0.7,0.8,0.9,1.1,1.2,1.3]
    blur_list = [2,3,4,5,6,7]
    illu_list = [0.4,0.6,0.8,1.2,1.4,1.6]
    proj_list = [1,2,3,-1,-2,-3]

    mix_list = [str(p_rot)+"_"+str(p_scale)+"_"+str(p_blur)+"_"+str(p_illu)+"_"+str(p_proj) for p_rot,p_scale,p_blur,p_illu,p_proj in zip(rot_list,scale_list,blur_list,illu_list,proj_list)]#str(p_rot)+"_"+str(p_scale)+"_"+str(p_illu)+"_"+str(p_blur)
    transform_params_list = [rot_list,scale_list,blur_list,illu_list,proj_list,mix_list]
    methods_list = ["SuperPoint","ORB","AGAST_SIFT","GFTT_SIFT"]
    transform_list = ["rot","scale","blur","illu","proj","mix"]




    all_repeatibility_dict = {"rot":[],"scale":[],"illu":[],"blur":[],"proj":[],"mix":[]}  # in the order of M methods  rot:[M values]  ...
    all_avg_loc_distance_dict = {"rot":[],"scale":[],"illu":[],"blur":[],"proj":[],"mix":[]}
    all_Homo_error_dict = {"rot":[],"scale":[],"illu":[],"blur":[],"proj":[],"mix":[]}
    all_Matching_accuracy_dict = {"rot":[],"scale":[],"illu":[],"blur":[],"proj":[],"mix":[]}

    nbr_trd = 5  #  manually change this from 1 to 5 to get five rounds of different hyperparameter
    get_metrics()
#why below code didn't work preperly as expected?
    # for i in nbr_trd_list:
    #     nbr_trd = i
    #     get_metrics()



