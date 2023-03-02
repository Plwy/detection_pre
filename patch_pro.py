import cv2
import random
import os
import numpy as np
import json
from tqdm import tqdm

"""
遍历训练文件夹train_img：
    从patch文件夹随机0-10张图
    随机位置坐标,
    贴图, 与已有roi不重叠, 限制n个循环后 找不到不重叠的。
同样方法遍历 val， test文件夹
"""

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


def is_overlapping(roi1, roi2):
    """
        判断两个roi框是否有交集        
    """
    flag = False
    iou = compute_IOU(roi1, roi2)
    if iou != 0:
        flag = True
    return flag


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def random_patch_img(patch_img, target_img, t_roi_list=None):
    """
        target_img上随机贴上patch_img
        patch_img 与t_roi_list原始的roi框计算  iou
    """
    patch_h, patch_w, c = patch_img.shape
    target_h, target_w, _ = target_img.shape

    f = False # 标记是否随机贴图成功
    if t_roi_list is None: 
        px = random.randrange(0, target_w - patch_w + 1)
        py = random.randrange(0, target_h - patch_h + 1)
        
        # 贴图
        target_img[py:py+patch_h, px:px+patch_w,:] = patch_img
    else:
        max_time = 3
        time = 1
        while(time <= max_time):
            px = random.randrange(0, target_w - patch_w + 1)
            py = random.randrange(0, target_h - patch_h + 1)
            p = (px, py, px+patch_w, py+patch_h)
            time += 1

            if len(t_roi_list) == 0:
                break

            i = 0
            for roi in t_roi_list:
                ol = compute_IOU(p, roi)
                if ol != 0: break     # 重叠则继续找点
                i += 1

            if i < len(t_roi_list):
                continue
            else:
                break
            
        # 超过循环数，未找到贴图位置，返回未贴图标志
        if time > max_time:
            return target_img, px, py, f
        else:
            target_img[py:py+patch_h, px:px+patch_w,:] = patch_img
            f = True
        return target_img, px, py, f


def get_all_roi(json_path):
    roi_list = []
    with open(json_path, 'r') as f:
        # 解析json
        info = json.load(f)
        roi_list = info["shapes"]
        for roi in roi_list:
            label = roi["label"]
            # 读json中的roi点， 左下点，右上点
            point = roi["points"]
            x1,y1= point[0]
            x2,y2 = point[1]
            x_min = min(x1,x2)
            x_max = max(x1,x2)
            y_min = min(y1,y2)
            y_max = max(y1,y2)
            roi = (x_min,y_min, x_max, y_max)
            roi_list.append(roi)
    return roi_list


def patch_main():
    """
        需要设置一张图的最大贴图数， 不考虑重叠的贴图
    """
    target_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_test/img"
    label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_test/label"
    patch_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/croped_imgs"
    save_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/test/patched_dir"

    targe_list = os.listdir(target_dir)
    sub_patch_dir_list = os.listdir(patch_dir)
    max_time = 10   # 每张图最大patch数
    prob_thread = 0.2  # 得到target后不贴的概率

    # 决定了标签 id，慎重设置
    label_all = ['VDa-', 'Rg-', 'Vb-', 'Cc-', 'Ca-', 
                'Nc-', 'Na-', 'Va-', 'VDb-', 'Vd-', 
                'VDc-', 'Ra-', 'Rb-', 'Vc-', 'Nb-', 'Rf-', 'Rh-']
    label_all.extend(sub_patch_dir_list)
    print(label_all)
    label_all_dic = {}
    for i, label in enumerate(label_all):
        label_all_dic[label] = i    
    print(label_all_dic)


    count = 0   # 记录贴图过的数据数 记录贴图过的数据数
    for target in tqdm(targe_list):
        # Minicopy的图不贴
        if target.find("Mini") != -1:
            continue
        
        # 贴还是不贴的概率
        prob_thread = 0.2
        prob = np.random.rand()
        if prob < prob_thread:
            continue
        else:
            count += 1
            target_path = os.path.join(target_dir,target)
            target_img = cv2.imread(target_path)
            h,w,_ = target_img.shape

            # 重写该target下的label
            target_name = os.path.splitext(target)[0]
            t_label_path = os.path.join(label_dir,target_name+'.txt')

            # # 不允许重叠，读json得到所有已有的roi框
            # t_json_path = os.path.join(json_dir,target_name+'.json')
            # t_roi_list = get_all_roi(t_json_path)

            # 追加新的roi
            t_label_txt = open(t_label_path, 'a')

            # 随机贴至多max_time张patch
            patched_img = target_img.copy()
            time = random.randrange(0, max_time)

            for i in range(time):
                # 每次选patch随机一个标签
                label_patch = random.randrange(0, len(sub_patch_dir_list))
                label = sub_patch_dir_list[label_patch]   
                patch_label_dir = os.path.join(patch_dir,label)
                patch_label_list = os.listdir(patch_label_dir)
                # 随机一个标签下的patch
                k = random.randrange(0, len(patch_label_list)) 
                patch_filename = patch_label_list[k]
                patch_file_path = os.path.join(patch_label_dir, patch_filename)
                patch_img = cv2.imread(patch_file_path)

                # 贴图不与已有元件重叠
                # patched_img, px, py, pw, ph = random_patch_img(patch_img, patched_img, t_roi_list)   #贴图
                patched_img, px, py, pw, ph = random_patch_img(patch_img, patched_img)   #允许重叠贴图

                # # 不允许重叠，添加新增的roi
                # new_roi = (px, py, px+pw, py+ph)
                # t_roi_list.append(new_roi)

                # 计算转yolo
                x_min = px
                x_max = px+pw
                y_min = py
                y_max = py+ph
                b = (float(x_min), float(x_max), float(y_min), float(y_max))
                b1, b2, b3, b4 = b
                # 标注越界修正
                if b2 > w:
                    b2 = w
                if b4 > h:
                    b4 = h
                b = (b1, b2, b3, b4)
                bb = convert((w, h), b)

                if label in label_all_dic:
                    cls_id = label_all_dic[label]

                # 追加进label
                t_label_txt.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            
            # 保存贴图后的图像
            save_path = os.path.join(save_dir, target)
            cv2.imwrite(target_path, patched_img)
            t_label_txt.close()

    print("patched img num:", count)

def random_patch_img_test():
    t_img = cv2.imread("/home/zsl/Music/1clip2.jpg")
    p_img = cv2.imread("/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/croped_imgs/C_G/C_G_1.jpg")

    patched_img, px, py = random_patch_img(p_img, t_img)
    cv2.imshow('patched_img', patched_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    patch_main()
    # random_patch_img_test()