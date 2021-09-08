import numpy as np
import sys
import matplotlib.pyplot as plt



#methods_list = ["SuperPoint","ORB","GFTT_SIFT","AGAST_SIFT"]
#all_repeatibility_dict = {"rot":[in the order of ["SuperPoint","ORB","GFTT_SIFT","AGAST_SIFT"]],"scale":[],"illu":[],"blur":[],"mix":[]}
def visulisation_certain_transform_across_methods(new_img_dir,transform,metric_name):

    data_across_methods =  {0:[],1:[],2:[],3:[]}#{"SuperPoint":[],"ORB":[],"GFTT_SIFT":[],"AGAST_SIFT":[]}
    for nbr_trd in [1,2,3,4,5]:
        rep_path = new_img_dir+"all_repeatibility_dict_trd"+str(nbr_trd)+".npz"
        # LE_path = new_img_dir+"all_avg_loc_distance_dict_trd"+str(nbr_trd)+".npz"
        # HE_path = new_img_dir+"all_Homo_error_dict_trd"+str(nbr_trd)+".npz"
        # MA_path = new_img_dir+"all_Matching_accuracy_dict_trd"+str(nbr_trd)+".npz"
        #
        rep_dict = np.load(rep_path)[transform].tolist()
        # LE_dict = np.load(LE_path)[transform].tolist()
        # HE_dict = np.load(HE_path)[transform].tolist()
        # MA_dict = np.load(MA_path)[transform].tolist()


        ##methods_list = ["SuperPoint","ORB","GFTT_SIFT","AGAST_SIFT"]
        for method_idx in range(4):
            # print("haha")
            data_across_methods[method_idx].append(rep_dict[method_idx])
        # x = range(len(data))
        # plt.plot(x,data['close'])
    for method_idx in range(len(data_across_methods)):
        x = range(1,6)  #  nbr_trd  1,2,3,4,5,
        plt.plot(x,data_across_methods[method_idx],label = str(method_idx))
    plt.legend()
    plt.show()

new_img_dir = r"E:\Datasets\surgical\out_imgs\\"
transform = "rot"
visulisation_certain_transform_across_methods(new_img_dir,transform,"metric_name")
