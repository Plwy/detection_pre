import numpy as np
import cv2
import os
import numpy as np
import random
import pywt
import pywt.data

import matplotlib.pyplot as plt

def imadjust(img, in_range=[0, 1], out_range=[0, 1], gamma=1):
    # 默认参数和matlab中的默认参数相同

    low_in, high_in = in_range[0], in_range[1]
    low_out, high_out = out_range[0], out_range[1]
    # 百分比截断
    p1, p99 = np.percentile(img, (1, 99))
    img_out = np.clip(img, p1, p99)
    img_out = (((img_out - low_in) / (high_in - low_in)) ** gamma) * (high_out - low_out) + low_out

    return img_out

import numpy as np

def imadjust2(img, out_range=[0, 1], gamma=1):
    # 默认参数和matlab中的默认参数相同
	
	# 百分比截断
    p1, p99 = np.percentile(img, (1, 99))
    img_out = np.clip(img, p1, p99)
    
    low_in, high_in = np.min(img), np.max(img)
    low_out, high_out = out_range[0], out_range[1]
    img_out = (((img_out - low_in) / (high_in - low_in)) ** gamma) * (high_out - low_out) + low_out

    return img_out


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)
 


def get_defect_main():
    """
        读入样板图和测试图，基于模板的缺陷检测。
    """
    mode = 6

    # imgt_p = "/media/zsl/data/workspace/codes/obj_detection/detection_pre/test/pair/多件样板.jpg"
    # # imgd_p = "/media/zsl/data/workspace/codes/obj_detection/detection_pre/test/pair/aglined_duojian.png"
    # imgt_p = "test/zsl_test_image/template/1.jpg"
    # imgd_p = "test/zsl_test_image/aligned/多件1_1_aligned.jpg"

    imgt_p = 'test/highlight_test/highlight0_result/xiufu_1.jpg'
    imgd_p = 'test/highlight_test/highlight0_result/xiufu_多件1_0_aligned.jpg'


    # # 成对的元件切图
    # imgt_p = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/all_pairs/diff1/d/裸板-B001_aligned_0_88.jpg"
    # imgd_p = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/all_pairs/diff1/t/裸板-B001_88.jpg"


    imgt = cv2.imread(imgt_p)
    imgd = cv2.imread(imgd_p)

    img_gama = imadjust(imgt)
    cv2.imwrite('gamat.png', img_gama)

    img_brighter = adjust_gamma(imgd, 2)
    cv2.imwrite('gammad_2.png')


if __name__ == "__main__":
    # absdiff()
    # template_match()
    get_defect_main()


