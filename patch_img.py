import os
import shutil
import cv2
import random
import numpy as np
import json
from tqdm import tqdm

# 
high = ["Ca", "Ra"]   #10000-15000  
high_l = ["Rh"]   #1000-5000    x5
mid = ["Na", "Va", "VDb"]    #500-1000   x10
low_h = ["Cc", "Nb", "Rb", "Vb", "Vc", "VDa"]  # 100-500   x 20
low = ["Nc", "Rf", "Rg", "VDc"]  # 0-100   x 50


"""
贴图方案：
1.一张图上重复贴一种大图，不可重叠，不考虑原始框，直到贴不下。
2. 一张图上重复贴一种小图， 不可重叠，考虑原始的大图roi,直到贴不下。 

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


def is_conclude(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 
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
        
        ## 几乎包括
        if S_cross/S1 > 0.9 or S_cross/S2 > 0.9:
            return 1
        else:
            return 0
        

def computer_IOU_conclude(rec1,rec2):
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
    is_conclude = False   # 是否为包括关系

    #两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0, is_conclude
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        IOU = S_cross/(S1+S2-S_cross)

        if S_cross/S1 > 0.9 or S_cross/S2 > 0.9:
            is_conclude = True
            return IOU, is_conclude
        else:
            return IOU, is_conclude





def random_patch_img(patch_img, target_img, t_roi_list=None):
    """
        target_img上随机贴上patch_img
        patch_img 与t_roi_list原始的roi框计算  iou
    """
    patch_h, patch_w, _ = patch_img.shape
    target_h, target_w, _ = target_img.shape

    f = False # 标记是否随机贴图成功
    if t_roi_list is None: 
        px = random.randrange(0, target_w - patch_w + 1)   # 随机左上角坐标
        py = random.randrange(0, target_h - patch_h + 1)    
        # 贴图
        target_img[py:py+patch_h, px:px+patch_w,:] = patch_img

        return target_img
    
    else: # 需要计算与原始roi框的重叠
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
                if ol != 0: break       # 重叠则继续找点
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

def get_all_roi(path):
    """
        读txt的标签得到roi框
        返回：{"label1":[左上角点，右下角点]， label2...}
    """
    rois = {}
    for line in open(path,"r"): #设置文件对象并读取每一行文件
        line = line.strip()
        x = line.split(' ')
        rois[x[0]] =(float(x[1]), float(x[2]), float(x[3]), float(x[4]))
    
    return rois

def patch_main1():
    """
    1.一张图上重复贴一种大图，不可重叠，不考虑原始框，直到贴不下。
    """
    target_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/PCB_data_base_nonoise"
    
    # label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/label_base_nonoise"
    patch_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/to_patch_dir"
    # 贴图后新图,新标签的保存目录
    patched_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_1/patched_img"
    patched_label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_1/patched_label"

    # 没有就创建
    if not os.path.exists(patched_dir):
        os.makedirs(patched_dir)
    if not os.path.exists(patched_label_dir):
        os.makedirs(patched_label_dir)

    targe_list = os.listdir(target_dir)
    sub_patch_dir_list = os.listdir(patch_dir)
    just_less = ['Ca-', 'Ra-', 'Rh-', 'VDb-']
    sub_patch_dir_list = [x for x in sub_patch_dir_list if x not in just_less]

    patch_record = {}  # 记录各个标签的贴图量
    for patch_label in sub_patch_dir_list:
        patch_record[patch_label] = 0
    

    target_num = 5000 # 每个贴图的目标贴图个数
    limit = 3   # 基础图4567 。至多循环贴3次以达到预定目标
    loop = 0
    p_c = 0
    while(loop < limit):
        # 判断哪些切图已经够了。
        for k in patch_record:
            if patch_record[k] > target_num:   # 切图满10000
                if k in sub_patch_dir_list:
                    sub_patch_dir_list.remove(k)
            
        # 待贴图都足够了
        if len(sub_patch_dir_list) == 0:
            print('all enough!')
            break

        # # base文件夹子循环几次来贴图
        # patched_sub_dir = os.path.join(patched_dir, str(loop))
        # if not os.path.exists(patched_sub_dir):
        #     os.makedirs(patched_sub_dir)

        # pl_sub_dir = os.path.join(patched_label_dir, str(loop))
        # if not os.path.exists(pl_sub_dir):
        #     os.makedirs(pl_sub_dir)

        loop += 1
        for target in tqdm(targe_list):
            target_path = os.path.join(target_dir, target)
            target_img = cv2.imread(target_path)
            h,w,_ = target_img.shape
            target_name = os.path.splitext(target)[0]
            img_ext = os.path.splitext(target)[-1]

            # 创建新的roi,因为不需要和原label中的roi做比较
            t_roi_list = []
            # t_txt_path = os.path.join(pl_sub_dir, target_name+'.txt')
            t_txt_path = os.path.join(patched_label_dir, target_name+'_'+str(loop)+'.txt')
            t_label_txt = open(t_txt_path, 'w')

            patched_img = target_img.copy()  # 待贴目标图

            # 随机一个标签
            label_patch = random.randrange(0, len(sub_patch_dir_list))
            p_label = sub_patch_dir_list[label_patch]   

            # 打开该标签目录进行贴图
            p_label_dir = os.path.join(patch_dir, p_label)
            p_label_list = os.listdir(p_label_dir)
            random.shuffle(p_label_list)   # 打乱

            # 遍历该标签的目录
            for p_label_file in p_label_list:
                p_label_path = os.path.join(p_label_dir, p_label_file)
                patch_img = cv2.imread(p_label_path)
                ph, pw, _ = patch_img.shape

                # 贴图
                patched_img, px, py, f = random_patch_img(patch_img, patched_img, t_roi_list)   
                # cv2.imshow('patched',patched_img)
                # cv2.waitKey(0)
                # 贴图成功
                if f:         
                    # 更新t_roi_list       
                    new_roi = (px, py, px+pw, py+ph)
                    t_roi_list.append(new_roi)

                    # 新框写入 txt
                    t_label_txt.write(p_label + " " + " ".join([str(a) for a in new_roi]) + '\n')

                    patch_record[p_label] += 1  # 切图种类计数
                    p_c += 1    # 总切图计数
                else:  # 贴图失败， 该patched_img 框满了
                    break 

            # 保存贴图后的图像
            # save_path = os.path.join(patched_sub_dir, target)
            save_path = os.path.join(patched_dir, target_name+'_'+str(loop)+img_ext)

            # cv2.imshow('patched',patched_img)
            # cv2.waitKey(0)
            cv2.imwrite(save_path, patched_img)
            t_label_txt.close()

    print("patched img num:", p_c)
    print(patch_record)

    return patch_record


def patch_main2(patch_record):
    """
    2.一张图上重复贴一种小图， 不可重叠，考虑原始的大图roi,直到贴不下。
    保存方式为 dir/loop/filename
    """
   
    target_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_1/patched_img"
    label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_1/patched_label"
    # target_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_test/img"
    # label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_test/label"
    patch_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/to_patch_dir"
    # 贴图后新图,新标签的保存目录
    patched_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_1/patched_img_2"
    patched_label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_1/patched_label_2"

    # 没有就创建
    if not os.path.exists(patched_dir):
        os.makedirs(patched_dir)
    if not os.path.exists(patched_label_dir):
        os.makedirs(patched_label_dir)

    targe_list = os.listdir(target_dir)
    # sub_patch_dir_list = os.listdir(patch_dir)
    # sub_patch_dir_list = ['Ca-', 'Ra-', 'Rh-', 'VDb-']
    # sub_patch_dir_list = [ 'Ra-',  'VDb-']
    sub_patch_dir_list = os.listdir(patch_dir)
    just_less = ['Ca-', 'Ra-', 'Rh-', 'VDb-']
    sub_patch_dir_list = [x for x in sub_patch_dir_list if x not in just_less]
    print("patch_dir: ",sub_patch_dir_list)

    # patch_record = {}  # 记录各个标签的贴图量
    # for patch_label in sub_patch_dir_list:
    #     patch_record[patch_label] = 0
    # print(patch_record)

    
    loop = 0
    limit = 1
    p_c = 0
    t_c = 0  # 处理 t的数数
    target_num = 5000
    while(loop < limit):
        # 判断哪些切图已经够了。
        for k in patch_record:
            if patch_record[k] > target_num:   # 切图满10000
                if k in sub_patch_dir_list:
                    sub_patch_dir_list.remove(k)
            
        # 待贴图都足够了
        if len(sub_patch_dir_list) == 0:
            print('all enough!')
            break

        # # base文件夹子循环几次来贴图
        # patched_sub_dir = os.path.join(patched_dir, str(loop))
        # if not os.path.exists(patched_sub_dir):
        #     os.makedirs(patched_sub_dir)

        # pl_sub_dir = os.path.join(patched_label_dir, str(loop))
        # if not os.path.exists(pl_sub_dir):
        #     os.makedirs(pl_sub_dir)

        loop += 1
        for target in tqdm(targe_list):
            t_c += 1
            target_path = os.path.join(target_dir, target)
            target_img = cv2.imread(target_path)
            h,w,_ = target_img.shape
            target_name = os.path.splitext(target)[0]
            img_ext = os.path.splitext(target)[-1]


            
            # # 因为创建新的roi,不需要和原label中的roi做比较
            # t_roi_list = []
            # # t_txt_path = os.path.join(pl_sub_dir, target_name+'.txt')
            # t_txt_path = os.path.join(patched_label_dir, target_name+'_'+str(loop)+'.txt')
            # t_label_txt = open(t_txt_path, 'w')

            # 在原始标签上行追加
            t_label_path = os.path.join(label_dir, target_name+'.txt')

            with open(t_label_path, 'r') as f:  
                r = []
                a = f.readlines()
                for l in a:
                    l = l.strip()
                    x = l.split(' ')
                    ll =(x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]))
                    r.append(ll)
                f.close()


            # 不追加，重写， 包括已有的框
            patched_label_path = os.path.join(patched_label_dir, target_name+'.txt')
            t_label_txt = open(patched_label_path, 'w')
            t_roi_list = []
            for k in r:
                t_roi_list.append(k[1:])
                t_label_txt.write(k[0] + " " + " ".join([str(a) for a in k[1:]]) + '\n')

            patched_img = target_img.copy()  # 待贴目标图

            # 随机一个标签
            label_patch = random.randrange(0, len(sub_patch_dir_list))
            p_label = sub_patch_dir_list[label_patch]   

            # 打开该标签目录进行贴图
            p_label_dir = os.path.join(patch_dir, p_label)
            p_label_list = os.listdir(p_label_dir)
            random.shuffle(p_label_list)   # 打乱

            # 遍历该标签的目录
            for p_label_file in p_label_list:
                p_label_path = os.path.join(p_label_dir, p_label_file)
                patch_img = cv2.imread(p_label_path)
                ph, pw, _ = patch_img.shape

                # 贴图
                patched_img, px, py, f = random_patch_img(patch_img, patched_img, t_roi_list)   
                # print('after:',len(t_roi_list))

                # cv2.imshow('patched',patched_img)
                # cv2.waitKey(0)
                # 贴图成功
                if f:         
                    # 更新t_roi_list       
                    new_roi = (px, py, px+pw, py+ph)
                    t_roi_list.append(new_roi)

                    # 新框写入 txt
                    t_label_txt.write(p_label + " " + " ".join([str(a) for a in new_roi]) + '\n')

                    patch_record[p_label] += 1  # 切图种类计数
                    p_c += 1    # 总切图计数
                else:  # 贴图失败， 该patched_img 框满了
                    break 

            # 保存贴图后的图像
            # save_path = os.path.join(patched_sub_dir, target)
            # save_path = os.path.join(patched_dir, target_name+'_'+str(loop)+img_ext)
            save_path = os.path.join(patched_dir, target_name +img_ext)

            # cv2.imshow('patched',patched_img)
            # cv2.waitKey(0)
            cv2.imwrite(save_path, patched_img)
            t_label_txt.close()

    print("patched img num:", p_c)
    print('target process num:', t_c)
    print(patch_record)

    return patch_record

def patch_main3(patch_record):
    """
    2.一张图上重复贴一种小图， 不可重叠，考虑原始的大图roi,直到贴不下。
    保存方式为 dir/loop/filename
    """
   
    target_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_1/patched_img_2"
    label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_1/patched_label_2"
    # target_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_test/img"
    # label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_test/label"
    patch_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/to_patch_dir"
    # 贴图后新图,新标签的保存目录
    patched_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_1/patched_img_mini"
    patched_label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/patch_1/patched_label_mini"

    # 没有就创建
    if not os.path.exists(patched_dir):
        os.makedirs(patched_dir)
    if not os.path.exists(patched_label_dir):
        os.makedirs(patched_label_dir)

    targe_list = os.listdir(target_dir)
    # sub_patch_dir_list = os.listdir(patch_dir)
    sub_patch_dir_list = ['Rh-', 'VDb-']
    # sub_patch_dir_list = [ 'Ra-',  'VDb-']

    # sub_patch_dir_list = os.listdir(patch_dir)
    # just_less = ['Ca-', 'Ra-', 'Rh-', 'VDb-']
    # sub_patch_dir_list = [x for x in sub_patch_dir_list if x not in just_less]
    # print("patch_dir: ",sub_patch_dir_list)

    patch_record = {}  # 记录各个标签的贴图量
    for patch_label in sub_patch_dir_list:
        patch_record[patch_label] = 0
    print(patch_record)

    
    loop = 0
    limit = 1
    p_c = 0
    t_c = 0  # 处理 t的数数
    target_num = 5000
    while(loop < limit):
        # 判断哪些切图已经够了。
        for k in patch_record:
            if patch_record[k] > target_num:   # 切图满10000
                if k in sub_patch_dir_list:
                    sub_patch_dir_list.remove(k)
            
        # 待贴图都足够了
        if len(sub_patch_dir_list) == 0:
            print('all enough!')
            break

        # # base文件夹子循环几次来贴图
        # patched_sub_dir = os.path.join(patched_dir, str(loop))
        # if not os.path.exists(patched_sub_dir):
        #     os.makedirs(patched_sub_dir)

        # pl_sub_dir = os.path.join(patched_label_dir, str(loop))
        # if not os.path.exists(pl_sub_dir):
        #     os.makedirs(pl_sub_dir)

        loop += 1
        for target in tqdm(targe_list):

        # 判断哪些切图已经够了。
            for k in patch_record:
                if patch_record[k] > target_num:   # 切图满10000
                    if k in sub_patch_dir_list:
                        sub_patch_dir_list.remove(k)
            if len(sub_patch_dir_list) == 0:
                print('all enough!')
                break           

            t_c += 1
            target_path = os.path.join(target_dir, target)
            target_img = cv2.imread(target_path)
            h,w,_ = target_img.shape
            target_name = os.path.splitext(target)[0]
            img_ext = os.path.splitext(target)[-1]


            
            # # 因为创建新的roi,不需要和原label中的roi做比较
            # t_roi_list = []
            # # t_txt_path = os.path.join(pl_sub_dir, target_name+'.txt')
            # t_txt_path = os.path.join(patched_label_dir, target_name+'_'+str(loop)+'.txt')
            # t_label_txt = open(t_txt_path, 'w')

            # 在原始标签上行追加
            t_label_path = os.path.join(label_dir, target_name+'.txt')

            with open(t_label_path, 'r') as f:  
                r = []
                a = f.readlines()
                for l in a:
                    l = l.strip()
                    x = l.split(' ')
                    ll =(x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]))
                    r.append(ll)
                f.close()


            # 不追加，重写， 包括已有的框
            patched_label_path = os.path.join(patched_label_dir, target_name+'.txt')
            t_label_txt = open(patched_label_path, 'w')
            t_roi_list = []
            for k in r:
                t_roi_list.append(k[1:])
                t_label_txt.write(k[0]+ " " + " ".join([str(a) for a in k[1:]]) + '\n')

            patched_img = target_img.copy()  # 待贴目标图

            # 随机一个标签
            label_patch = random.randrange(0, len(sub_patch_dir_list))
            p_label = sub_patch_dir_list[label_patch]   

            # 打开该标签目录进行贴图
            p_label_dir = os.path.join(patch_dir, p_label)
            p_label_list = os.listdir(p_label_dir)
            random.shuffle(p_label_list)   # 打乱

            # 遍历该标签的目录
            for p_label_file in p_label_list:
                p_label_path = os.path.join(p_label_dir, p_label_file)
                patch_img = cv2.imread(p_label_path)
                ph, pw, _ = patch_img.shape

                # 贴图
                patched_img, px, py, f = random_patch_img(patch_img, patched_img, t_roi_list)   
                # print('after:',len(t_roi_list))

                # cv2.imshow('patched',patched_img)
                # cv2.waitKey(0)
                # 贴图成功
                if f:         
                    # 更新t_roi_list       
                    new_roi = (px, py, px+pw, py+ph)
                    t_roi_list.append(new_roi)

                    # 新框写入 txt
                    t_label_txt.write(p_label + " " + " ".join([str(a) for a in new_roi]) + '\n')
                    if p_label in patch_record:

                        patch_record[p_label] += 1  # 切图种类计数
                    else:
                        patch_record[p_label] = 1
                    p_c += 1    # 总切图计数
                else:  # 贴图失败， 该patched_img 框满了
                    break 

            # 保存贴图后的图像
            # save_path = os.path.join(patched_sub_dir, target)
            # save_path = os.path.join(patched_dir, target_name+'_'+str(loop)+img_ext)
            save_path = os.path.join(patched_dir, target_name +img_ext)

            # cv2.imshow('patched',patched_img)
            # cv2.waitKey(0)
            cv2.imwrite(save_path, patched_img)
            t_label_txt.close()

    print("patched img num:", p_c)
    print('target process num:', t_c)
    print(patch_record)

if __name__ == '__main__':
    # 1. 先贴大图
    patch_record = patch_main1()
    print("=========phse1========:")
    print(patch_record)
    patch_record = {'Cc-': 6133, 'Vc-': 5286, 'VDa-': 7784, 
                    'VDc-': 6678, 'Nc-': 4560, 'Rg-': 4146, 
                    'Nb-': 2952, 'Rb-': 6410, 'Na-': 3788, 
                    'Vb-': 2779, 'Va-': 7277, 'Rf-': 4805}
    # 再贴一次大图
    patch_record = patch_main2(patch_record)
    print("=========phse2========:")
    print(patch_record)

    # 再贴小图
    print("before:",patch_record)
    patch_record = patch_main3(patch_record)
    print("=========phse3========:")
    print(patch_record)

#{'Va-': 7277, 'Vc-': 5286, 'VDc-': 6678, 'Nc-': 3187, 'Rb-': 6410, 'Rg-': 5682, 'Na-': 5597, 'Vb-': 5818, 'Nb-': 6877, 'VDa-': 7784, 'Cc-': 6140, 'Rf-': 7328}