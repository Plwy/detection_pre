import os
import shutil
import cv2
import random
import numpy as np
import json
from tqdm import tqdm

from draw_labels import txt_parse
import glob


def txt_parse(txt_path):
    """
    parse .txt
    return label x1 y1 x2 y2
    """
    r = []
    with open(txt_path, 'r') as f:  
        a = f.readlines()
        for l in a:
            l = l.strip()
            x = l.split(' ')
            if len(x) < 5:
                t = []
                y = x[0].split('-')
                t.append(y[0]+'-')
                t.append(y[1])
                t.append(x[1])
                t.append(x[2])
                t.append(x[3])
                x = t           
            ll =[x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4])]
            r.append(ll)
        f.close()
    return r

def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)


def compute_ratio(rec1, rec2):

    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        # return S_cross/(S1+S2-S_cross)
        return S_cross/S1


def patch_issue_main():
    """
        在切图贴图后， 添加原图上未覆盖的标签
    """
    # 原图及标签
    target_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/PCB_data_base_nonoise"
    label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/label_base_nonoise_txt"
    # 贴图后的新图及标签
    patched_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/images/train_new_all"
    patched_label_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/labels/train_txt_label"

    # 计算roi后，更新标签
    real_p_label_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/labels/train_txt_label_real"
    # real_yolo_label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_test/patched_label_real_yolo"

    if not os.path.exists(real_p_label_dir):
        os.makedirs(real_p_label_dir, exist_ok=True)


    target_list = os.listdir(target_dir)
    patched_list = os.listdir(patched_dir)

    for target in tqdm(target_list):
        t_name = os.path.splitext(target)[0]
        t_ext = os.path.splitext(target)[1]
        t_label_path = os.path.join(label_dir,t_name+'.txt')

        t_roi_list = txt_parse(t_label_path)

        for patched in patched_list:
            p_name = os.path.splitext(patched)[0]
            p_ext = os.path.splitext(patched)[1]    
            
            # 找target对应的patch们， 
            if p_name.find(t_name) == 0:
                # print(t_name, p_name)
                # 重写
                real_p_label_path = os.path.join(real_p_label_dir,p_name+'.txt')
                label_real_txt = open(real_p_label_path, 'w')
                # 读贴图的label
                p_label_path = os.path.join(patched_label_dir, p_name+'.txt')
                p_roi_list = txt_parse(p_label_path)
                # 先写入切图的框
                for p_roi in p_roi_list:
                    p_label = p_roi[0]
                    p_box = p_roi[1:]                    
                    label_real_txt.write(p_label + " " + " ".join([str(a) for a in p_box]) + '\n')


                # 遍历每个t_roi 和 每个p_roi的交集
                thread = 0.5     # 丢弃框的覆盖率阈值
                for t_roi in t_roi_list:
                    t_label = t_roi[0]
                    t_box = t_roi[1:]
                    # 遍历每个堆叠的图，计算覆盖面积
                    ratio = 0
                    for p_roi in p_roi_list:
                        p_label = p_roi[0]
                        p_box = p_roi[1:]

                        # 累加覆盖占总面积的比例
                        ratio += compute_ratio(t_box, p_box)
                    if ratio > thread:   # 覆盖太多则该原始框不要了
                        continue
                    else:
                        label_real_txt.write(t_label + " " + " ".join([str(a) for a in t_box]) + '\n')
                        
                label_real_txt.close()



                

if __name__ == '__main__':
    patch_issue_main()