
import os
from tkinter import Y
import cv2
import time
import codecs
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
from tqdm import trange                  # 显示进度条
from multiprocessing import cpu_count    # 查看cpu核心数
from multiprocessing import Pool

from patch_img import compute_IOU
import random
import numpy as np
import glob

from xml2yolo import txt_parse
import torch

import torchvision

from stitch_test import crop_imgs, stitch_test
from draw_labels import box_label_yl5

"""
1.切图   
2.检测  
3.拼接(生成txt)
4.对比 
"""

def flow_main():
    """批量切图并保存
    """
    base_images_dir = '/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare2/defects_aligned_white'   # 原始的图片目录
    crops_dir = '/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare2/defects_aligned_white_crop_imgs' # 切好的图保存的目录
    crop_imgs(base_images_dir, crops_dir)    # 


def batch_stitch_template():
    """批量合并模板图的检测框
    crop_dir  原图切的图
    crop_annos_dir 原图切图后的检测结果
    anno_stitch_dir  拼接后的anno保存路径
    img_stitch_dir  拼接后的img 
    """
    img_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_imgs"
    crop_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_crop_imgs_withori"
    crop_annos_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_crop_imgs_withori_labels_6wbig_v7pre50"
    anno_stitch_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_anno_stitch_6wbig_v7pre5"
    img_stitch_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_img_stitch_6wbig_v7pre5"

    dir_withfile = []
    for cur, subs, files in os.walk(crop_dir):
        if len(files)==0:
            continue
        else:
            f_2,f_1 = cur.split('/')[-2], cur.split('/')[-1]
            os.makedirs(anno_stitch_dir+'/'+f_2, exist_ok=True)
            os.makedirs(img_stitch_dir+'/'+f_2 , exist_ok=True)
            dir_withfile.append([cur, f_2,f_1])
    
    #['/defects_aligned_croped_imgs/6/缺R27  多C99 2022052301_aligned', '6', '缺R27  多C99 2022052301_aligned']
    for d in dir_withfile:   # 遍历所有切图标注文件夹, 每个文件夹对应一个原图和 
        dir, f2, f1 = d    #  dir 为切图文件夹
        # ori_path = os.path.join(img_dir,f2,f1+'.jpg')
        ori_path = os.path.join(img_dir,f1+'.jpg')

        crops_anno_dir=os.path.join(crop_annos_dir,f2,f1,'labels')
        stitch_dir = os.path.join(anno_stitch_dir,f2)
        detect_dir = os.path.join(img_stitch_dir,f2)

        print(ori_path)
        print(crops_anno_dir)
        print(stitch_dir)
        print(detect_dir)
        print("===")

        stitch_test(crops_anno_dir, ori_path, stitch_dir, detect_dir)

def batch_stitch_defect():
    """批量合并模板图的检测框
    img_dir 原图
    crop_dir  原图切的图
    crop_annos_dir 原图切图后的检测结果
    anno_stitch_dir  拼接后的anno保存路径
    img_stitch_dir  拼接后的img 
    """
    img_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white"
    crop_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_crop_withori"
    crop_annos_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_crop_withori_labels_6wbig79"
    anno_stitch_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_stitch_anno_6wbig79"
    img_stitch_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_stitch_img_6wbig79"

    
    # ## 测旋转的
    # img_dir = "/media/zsl/data/zsl_datasets/PCB_test/anno_merge_test/issues/rots/aug_imgs_rot"
    # crop_dir = "/media/zsl/data/zsl_datasets/PCB_test/anno_merge_test/issues/rots/crop_imgs"
    # crop_annos_dir = "/media/zsl/data/zsl_datasets/PCB_test/anno_merge_test/issues/rots/crops_labels"
    # anno_stitch_dir = "/media/zsl/data/zsl_datasets/PCB_test/anno_merge_test/issues/rots/stitch_anno_705"
    # img_stitch_dir = "/media/zsl/data/zsl_datasets/PCB_test/anno_merge_test/issues/rots/stitch_img_705"

    dir_withfile = []
    for cur, subs, files in os.walk(crop_dir):
        if len(files)==0:
            continue
        else:
            f_2,f_1 = cur.split('/')[-2], cur.split('/')[-1]
            os.makedirs(anno_stitch_dir+'/'+f_2, exist_ok=True)
            os.makedirs(img_stitch_dir+'/'+f_2 , exist_ok=True)
            dir_withfile.append([cur, f_2,f_1])
    
    #['/defects_aligned_croped_imgs/6/缺R27  多C99 2022052301_aligned', '6', '缺R27  多C99 2022052301_aligned']
    for d in dir_withfile:   # 遍历所有切图标注文件夹, 每个文件夹对应一个原图和 
        dir, f2, f1 = d    #  dir 为切图文件夹

        # 双层目录  img/1/crop1/1_crop.jpg
        ori_path = os.path.join(img_dir,f2,f1+'.jpg')    # d
        # ori_path = os.path.join(img_dir,f1+'.jpg')    # t

        crops_anno_dir=os.path.join(crop_annos_dir,f2,f1,'labels')
        stitch_dir = os.path.join(anno_stitch_dir,f2)    
        detect_dir = os.path.join(img_stitch_dir,f2)

        # #单层目录
        # ori_path = os.path.join(img_dir,f1+'.jpg')
        # crops_anno_dir=os.path.join(crop_annos_dir,f1,'labels')
        # stitch_dir = os.path.join(anno_stitch_dir)
        # detect_dir = os.path.join(img_stitch_dir)

        print(ori_path)
        print(crops_anno_dir)
        print(stitch_dir)
        print(detect_dir)
        print("===")

        stitch_test(crops_anno_dir, ori_path, stitch_dir, detect_dir)



if __name__ == "__main__":
    # flow_main()
    # batch_stitch_template()
    batch_stitch_defect()