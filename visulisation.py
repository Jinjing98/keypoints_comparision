import numpy as np
import sys
import matplotlib.pyplot as plt




#all_repeatibility_dict = {"rot":[in the order of ["SuperPoint","ORB","GFTT_SIFT","AGAST_SIFT"]],"scale":[],"illu":[],"blur":[],"proj":[],"mix":[]}
def visulisation_certain_transform_across_methods(new_img_dir,transform,metric_term):

    for nbr_trd in nbr_trd_list:

        metric_path = new_img_dir+metric_term+str(nbr_trd)+".npz"
        metric_dict = np.load(metric_path)[transform].tolist()


        ##methods_list = ["SuperPoint","ORB","GFTT_SIFT","AGAST_SIFT"]
        for method_idx in range(len(methods_list)):
            piece_of_data = metric_dict[method_idx]
            data_across_methods[method_idx].append(piece_of_data)

    for method_idx,method_name in zip(range(len(data_across_methods)),methods_list):
        x = range(1,6)  #  nbr_trd  1,2,3,4,5,
        plt.plot(x,data_across_methods[method_idx], marker="*",label = str(method_name))
    plt.legend()
    plt.title(transform+" "+metric_term)
    plt.show()

new_img_dir = r"E:\Datasets\surgical\out_imgs\\"
nbr_trd_list = [1,2,3,4,5]
methods_list = ["SuperPoint","ORB","GFTT_SIFT","AGAST_SIFT"]
data_across_methods =  {0:[],1:[],2:[],3:[]}#{"SuperPoint":[],"ORB":[],"GFTT_SIFT":[],"AGAST_SIFT":[]}
metrics_terms_list = ["all_repeatibility_dict_trd",\
                    "all_avg_loc_distance_dict_trd",\
                    "all_Homo_error_dict_trd",\
                    "all_Matching_accuracy_dict_trd"]
transform_list = ["rot","scale","blur","illu","proj","mix"]



metric =  "all_avg_loc_distance_dict_trd"#"all_Matching_accuracy_dict_trd"#"all_Homo_error_dict_trd"#"all_avg_loc_distance_dict_trd"#"all_repeatibility_dict_trd"
transform = "mix"
visulisation_certain_transform_across_methods(new_img_dir,transform,metric)
