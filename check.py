
import glob
import os
import cv2
from tqdm import tqdm
import shutil

from compare import txt_parse
from stitch_test import box_label_yl5, make_slice_txt


def check_label():
    """
        检查标签是否正确,在图中画出标签
    """
    dir1 = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/1_big_obj_crops_imgs_augs"  # image
    dir2 = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/zzzz"  # label
    dir3 = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/444000check"
    list1 = os.listdir(dir1)
    print("====1")
    for fname in list1:
        print(fname)
        file1 = os.path.join(dir1, fname)
        file2 = os.path.join(dir2, fname.replace('.jpg','.txt'))

        objlist = txt_parse(file2)

        p = os.path.join(dir3, fname)
        img = cv2.imread(file1)
        img_ori = img.copy()
        for obj in objlist:
            box = obj[1:]
            img_ori = box_label_yl5(img_ori, box)
        
        cv2.imwrite(p,img_ori)




def check_txt():
    """
        检查txt中出现的少空格; 框左标出现负
        返回错误文本数，及路径
    """

    t_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/labels/train"
    # t_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/labels/label_base_nonoise_txt"

    x = glob.glob(t_path+"/*")
    c = []
    c_neg = []
    for i in tqdm(x):
        c1, c2 =  _check_txt(i)
        c.extend(c1)
        c_neg.extend(c2)

    c = list(set(c))
    c_neg = list(set(c_neg))

    print("space:",len(c))
    print(c)
    print("neg:",len(c_neg))
    print(c_neg)


def _check_txt(txt_path):
    """
    .txt
    label x1 y1 x2 y2
    """
    c = [] # 少” “问题
    c_neg = [] # 出现负值问题
    with open(txt_path, 'r') as f:  
        a = f.readlines()
        for l in a:
            l = l.strip()
            x = l.split(' ')
            if len(x) < 5:
                c.append(txt_path)
                break
        for l in a:
            l = l.strip()
            x = l.split(' ')
            for i in x[1:]:
                t = float(i)
                if t < 0:
                    c_neg.append(txt_path)
                    break
    return c, c_neg


def get_wrong_files(txt_path):
    with open(txt_path, 'r') as f:
        t = f.read().strip().splitlines()
    
    print(len(t))
    t = list(set(t))

    print(len(t))
    return t

def get_wrong_pics(wrong_list):
    base_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset/PCB_data_base"
    noise_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset/PCB_data_base_noise"
    base_list = os.listdir(base_dir)
    print("ori base len:", len(base_list))
    c = 0
    for base_file in tqdm(base_list):
        flag = False
        for s in wrong_list:   
            if base_file.find(s+'clip') < 0 and base_file.find(s+'resize') < 0 and base_file.find(s+'random') < 0:
                continue
            else:
                flag = True
                print(os.path.join(base_dir, base_file))
                break
        if flag:
            c  += 1
            src = os.path.join(base_dir, base_file)
            dis = os.path.join(noise_dir, base_file)
            shutil.move(src, dis)
        else:
            continue

    print("nonoise base len:", len(base_list)-c)


def get_wrong_file_main():
    """
    先遍历错误标签的文件编号。
    从base中移出错误的到指定文件夹
    """
    wrong_list = get_wrong_files("wrong.txt")
    get_wrong_pics(wrong_list)


if __name__ == "__main__":
    check_label()