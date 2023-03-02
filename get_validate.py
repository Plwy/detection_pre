import os
import cv2
import time
import codecs
import xml.etree.ElementTree as ET
# from torch import R
from tqdm import tqdm
import shutil
from tqdm import trange                  # 显示进度条
from multiprocessing import cpu_count    # 查看cpu核心数
from multiprocessing import Pool

from patch_img import computer_IOU_conclude, is_conclude
import random
import numpy as np

from PIL import Image, ImageDraw, ImageFont
# from pathlib import Path
from compare import txt_parse, _compare, compare_one


def compare_gt_label(label_gt_path, label_d_path):
    """
        比较gt和测试板的labels ,返回 recal Pre, F1
    """
    gt_comp_list = txt_parse(label_gt_path)
    d_comp_list = txt_parse(label_d_path)

    # img_d = cv2.imread(img_d_path)
    # img_t = cv2.imread(img_t_path)

    matchest_l, difflabel_l, misspart_l, extrapart_l, missalign_l = _compare(gt_comp_list, d_comp_list)
    if len(matchest_l) != len(gt_comp_list):

        print("gt box num:", len(gt_comp_list))
        print("d box num:", len(d_comp_list))
        print("total match num:", len(matchest_l))
        print("match but label not num:", len(difflabel_l))
        print("missPart num:", len(misspart_l))
        print("missalign num:", len(missalign_l))
        print("extraPart num:", len(extrapart_l))

    # precise
    pre = len(matchest_l)/len(d_comp_list)
    recall = len(matchest_l)/len(gt_comp_list)
    f1 = 2*pre*recall/(pre+recall)

    return pre, recall, f1

    
def template_main():
    gt_anno_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/test/template_gt_txt"
    test_anno_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_anno_stitch_new_v7tinyb/template"

    cnt = 0
    precise_all = 0
    recall_all = 0
    f1_all = 0
    for cur, sub, files in os.walk(gt_anno_dir):
        if len(files) == 0:
            continue
        else:
            for file in files:
                # if file.find('1.txt') < 0 or file.find('2.txt') < 0:
                #     print(file)
                gt_anno_path = os.path.join(cur, file)
                test_anno_path = os.path.join(test_anno_dir, file)
                precises, recall, f1 = compare_gt_label(gt_anno_path, test_anno_path)
                # print("====")
                print(gt_anno_path)
                print(precises, recall, f1)
                cnt += 1
                precise_all += precises
                recall_all += recall
                f1_all += f1
    print("total pics:", cnt)
    print("total precises:", precise_all/cnt)
    print("total precises:", precise_all/cnt)
    print("total precises:", precise_all/cnt)


def test_main():
    gt_anno_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/test/defects_aligned_white_labeled"
    test_anno_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_stitch_anno_6wbig_v722/"
    pic = []

    for cur, sub, files in os.walk(test_anno_dir):
        if len(files) == 0:
            continue
        else:
            pre_total = 0
            recall_total = 0
            f1_total = 0
            c = 0
            for file in files:
                if file.endswith('.txt'):
                    c += 1
                    print(cur, file)
                    f_2, f_1 = cur.split('/')[-2], cur.split('/')[-1]
                    test_anno_path= os.path.join(cur, file)
                    gt_anno_path = os.path.join(gt_anno_dir, f_1, file)

                    precises, recall, f1 = compare_gt_label(gt_anno_path, test_anno_path)

                    pre_total += precises
                    recall_total += recall
                    f1_total += f1
                    print(precises, recall, f1)

            print("total num:", c)
            print("precise mean:",pre_total/c)
            print("recall mean:",recall_total/c)
            print("f1 mean:",f1_total/c)
    print(pic)


def compare_gt():
    """
        测试图的检测结果和gt的对比可视化生成
    """
    label_gt_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_labeled"
    label_d_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_stitch_anno_6wbig79"
    img_d_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_stitch_img_6wbig79"
    img_gt_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_gt_img"

    save_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/test/defect_gt_compare_6wbig79"

    # label_gt_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_labeled"
    # label_d_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_anno_stitch_6wbig_v7pre5/template"
    # img_d_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_img_stitch_6wbig_v7pre5"
    # img_gt_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_gt_imgs"

    # save_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/test/template_gt_compare_6wbig_v7pre5"


    
    os.makedirs(save_dir, exist_ok=True)


    pre_total = 0
    recall_total = 0
    f1_total = 0
    c = 0
    for cur_dir, sub_dir, files in os.walk(label_d_dir):
        if len(files) == 0:
            continue
        else:
            t_name = cur_dir.split('/')[-1]


            for file in tqdm(files):
                c += 1
                d_name, d_ext = os.path.splitext(file)
                if d_name.find('_ori') >= 0:   
                    continue
                else:
                    # # test
                    img_d_path = os.path.join(img_d_dir, t_name, file.replace('.txt','_merge.jpg'))
                    img_gt_path = os.path.join(img_gt_dir, t_name, file.replace('.txt', '.jpg'))  # 
                    label_gt_path = os.path.join(label_gt_dir, t_name, d_name+'.txt')
                    label_d_path = os.path.join(label_d_dir, t_name, d_name+'.txt')
                    # temp
                    # img_gt_path = os.path.join(img_gt_dir,  file.replace('.txt', '.jpg'))  # 
                    # label_gt_path = os.path.join(label_gt_dir,  d_name+'.txt')
                    # label_d_path = os.path.join(label_d_dir,  d_name+'.txt')
                    print(label_gt_path)
                    print(label_d_path)
                    print(img_gt_path)
                    print(img_d_path)
                    print("====")

                    precises, recall, f1 = compare_gt_label(label_gt_path, label_d_path)
                    if precises*recall*f1 != 1:
                        img_compare_draw = compare_one(label_gt_path, label_d_path, img_gt_path, img_d_path)

                        imgd_name, imgd_ext = os.path.splitext(os.path.basename(img_d_path))
                        save_path = os.path.join(save_dir, imgd_name+"_all"+imgd_ext)
                        cv2.imwrite(save_path, img_compare_draw)



                    pre_total += precises
                    recall_total += recall
                    f1_total += f1
                    print(precises, recall, f1)

    print("total num:", c)
    print("precise mean:",pre_total/c)
    print("recall mean:",recall_total/c)
    print("f1 mean:",f1_total/c)



if __name__ == "__main__":
    # template_main()
    # test_main()
    compare_gt()