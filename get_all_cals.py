

import glob
import os
import cv2
import json
from tqdm import tqdm

import xml.etree.ElementTree as ET


"""
看看原始数据集中， 每种类别的组件图张什么样子，存储比较类别分布。
"""
def json_parse(json_path):
    with open(json_path, 'r') as f:
        # 解析json
        info = json.load(f)
        roi_list = info["shapes"]
    return roi_list


def parse2label(x):
    """
    {"1":200,"2":200} --> {"Ca":200, "Va":200}酱
    """
    label= {'Va-': 0, 'Vc-': 1, 'VDc-': 2, 
            'Nc-': 3, 'VDb-': 4, 'Ca-': 5, 
            'Rh-': 6, 'Rb-': 7, 'Rg-': 8, 
            'Na-': 9, 'Vb-': 10, 'Nb-': 11, 
            'Ra-': 12, 'VDa-': 13, 'Cc-': 14, 'Rf-': 15}
    # label = []

    # x ={'14': 5826, '9': 5429, '1': 5050, '8': 5017, '4': 5067, '2': 6019, '5': 12732, '3': 4424, '10': 3327, '6': 6668, '11': 3758, '13': 7229, '7': 5765, '12': 9289, '15': 6129, '0': 6877}

    label_list = []
    for i in label.keys():
        label_list.append(i)

    label_cls = {}
    print(label_list)
    print(x)
    for i in x:
        l = label_list[int(i)]
        label_cls[l] = x[i]

    return label_cls

def txt_parse_main():
    """
        处理txt标签。返回所有目标标签的类别及对应数量
    """
    txt_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/label_base_nonoise_txt"
    img_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/for_train/resize_base_nonoise"
    output_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/tt"

    img_lists = os.listdir(img_dir)


    label= {'Va-': 0, 'Vc-': 1, 'VDc-': 2, 
            'Nc-': 3, 'VDb-': 4, 'Ca-': 5, 
            'Rh-': 6, 'Rb-': 7, 'Rg-': 8, 
            'Na-': 9, 'Vb-': 10, 'Nb-': 11, 
            'Ra-': 12, 'VDa-': 13, 'Cc-': 14, 'Rf-': 15}

    c = 0
    limit = 100000
    roi_lists = []
    for img_name in tqdm(img_lists):
        # 或者 限制处理文件数
        if c > limit:
            break
        filename = os.path.splitext(os.path.basename(img_name))[0]
        ext = '.txt'
        label_path = os.path.join(txt_dir, filename + ext)
        if not os.path.exists(label_path):
            continue

        c += 1
        img_path = os.path.join(img_dir,img_name)
        img = cv2.imread(img_path)  # 用来切图
        # 解析每张图的label_path

        # roi_lists = []
        with open(label_path, 'r') as f:
            a = f.readlines()
            for l in a:
                l = l.strip()
                x = l.split(' ')
                roi = {}
                for k, v in label.items():
                    if v == x[0]: 
                        roi["label"] = k
                    else:
                        roi["label"] = None
                roi["points"] = x[1:]
                roi_lists.append(roi)

        f.close()

    x = {}
    for roi in roi_lists:
        label = roi["label"] 
        if label in x:
            x[label] += 1
        else:
            x[label] = 1

    label_cls = parse2label(x)
    print("cls:")
    print(label_cls)

        # # 切出并保存到文件夹
        # for roi in roi_lists:
        #     label_name = roi["label"]

        #     # 创建label对应的文件夹
        #     if not os.path.exists(os.path.join(output_dir, label_name)):
        #         os.makedirs(os.path.join(output_dir, label_name))
        #         num_ = 1
        #     else:
        #         num_ = len(os.listdir(os.path.join(output_dir, label_name))) + 1

        #     points = roi["points"]
        #     x1,y1,x2,y2= points
        #     x1,y1,x2,y2 = float(x1), float(y1), float(x2), float(y2)
        #     patch = img[int(y1):int(y2),int(x1):int(x2),:]

        #     save_path = os.path.join(output_dir, label_name,filename+"_"+str(num_)+".jpg")
        #     cv2.imwrite(save_path, patch)

def xml_parse_main():
    json_dir = "/media/zsl/data/zsl_datasets/for_train/scheme1/pcba_wav_base/labels"
    img_dir = "/media/zsl/data/zsl_datasets/for_train/scheme1/pcba_wav_base/img"
    output_dir = "/media/zsl/data/zsl_datasets/for_train/scheme1/pcba_wav_base/cls"
    img_lists = os.listdir(img_dir)
    ext = '.xml'

    c = 0
    limit = 100000
    for img_name in tqdm(img_lists):
        # 停止的限制条件，label_all的key都已经创建
        # 或者 限制处理文件数
        if c > limit:
            break
        filename = os.path.splitext(os.path.basename(img_name))[0]
        label_path = os.path.join(json_dir, filename + ext)
        if not os.path.exists(label_path):
            continue

        c += 1
        img_path = os.path.join(img_dir,img_name)
        img = cv2.imread(img_path)

        # 解析每张图的label_path
        tree = ET.parse(label_path)
        for obj in tree.findall('object'):
            label_name = obj.find('name').text
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)

            if not os.path.exists(os.path.join(output_dir, label_name)):
                os.makedirs(os.path.join(output_dir, label_name))
                num_ = 1
            else:
                num_ = len(os.listdir(os.path.join(output_dir, label_name))) + 1


            patch = img[y_min:y_max,x_min:x_max,:]

            save_path = os.path.join(output_dir, label_name,filename+"_"+str(num_)+".jpg")
            cv2.imwrite(save_path, patch)





def json_parse_main():
    # json_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset/labels_json"
    json_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/label_base_nonoise"
    img_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/for_train/resize_base_nonoise"
    output_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/for_train/resize_base_nonoise_cls"

    img_lists = os.listdir(img_dir)
    # label_all = []
    # label_all = ['VDa-', 'Rg-', 'Vb-', 'Cc-', 'Ca-', 
    #             'Nc-', 'Na-', 'Va-', 'VDb-', 'Vd-', 
    #             'VDc-', 'Ra-', 'Rb-', 'Vc-', 'Nb-', 'Rf-', 'Rh-']
    # label_all_dic = {}
    # for i, label in enumerate(label_all):
    #     label_all_dic[label] = i


    c = 0
    limit = 100000
    for img_name in tqdm(img_lists):
        # 停止的限制条件，label_all的key都已经创建

        # 或者 限制处理文件数
        if c > limit:
            break
        filename = os.path.splitext(os.path.basename(img_name))[0]
        ext = '.json'
        label_path = os.path.join(json_dir, filename + ext)
        if not os.path.exists(label_path):
            continue

        c += 1
        img_path = os.path.join(img_dir,img_name)
        img = cv2.imread(img_path)
        # 解析每张图的label_path
        roi_lists = json_parse(label_path)
        for roi in roi_lists:
            label_name = roi["label"]

            # 创建label对应的文件夹
            if not os.path.exists(os.path.join(output_dir, label_name)):
                os.makedirs(os.path.join(output_dir, label_name))
                num_ = 1
            else:
                num_ = len(os.listdir(os.path.join(output_dir, label_name))) + 1

            points = roi["points"]
            x1,y1= points[0]
            x2,y2 = points[1]
            x_min = int(min(x1,x2))
            x_max = int(max(x1,x2))
            y_min = int(min(y1,y2))
            y_max = int(max(y1,y2))

            patch = img[y_min:y_max,x_min:x_max,:]

            save_path = os.path.join(output_dir, label_name,filename+"_"+str(num_)+".jpg")
            cv2.imwrite(save_path, patch)


def json_parse(json_path):

    with open(json_path, 'r') as f:
        # 解析json
        info = json.load(f)
        roi_list = info["shapes"]
        h = info["imageHeight"]
        w = info["imageWidth"]

        # for roi in roi_list:
        #     label = roi["label"]
        #     roi_dic[label] = roi["points"]
        #     print(roi)

        #     roi_list.append(roi_dic)
    f.close()
    return roi_list

def get_cls_fdir():
    """
        已经分好的类别的文件夹返回 ，类别及切图数
    """
    cls_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset/PCB_data_base_nonoise_patch_cls"
    l1 = os.listdir(cls_dir)
    cls_dic = {}
    for l in l1:
        sub_dir = os.path.join(cls_dir, l)
        l2 = os.listdir(sub_dir)
        num = len(l2)
        cls_dic[l] = num
    print(cls_dic)

if __name__ == "__main__":
    # json_parse_main()
    xml_parse_main()
    # txt_parse_main()
    # get_cls_fdir()