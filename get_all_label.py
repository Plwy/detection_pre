import glob
import os
import cv2
import json
from tqdm import tqdm


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


def json2txt_test(json_path, txt_path):
    roi_list = []
    points = []
    with open(json_path, 'r') as f:
        # 开始写txt
        label_txt = open(txt_path, "w")
        # 解析json
        info = json.load(f)
        roi_list = info["shapes"]
        h = info["imageHeight"]
        w = info["imageWidth"]
        for roi in roi_list:
            # 读json中的roi点， 左下点，右上点
            point = roi["points"]
            points.append(point)

    return roi_list, points


def json_parse_main():
    """
        json到到所有类别标签个数
    """
    json_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset/labels_json"
    json_files = os.listdir(json_dir)

    label_all = []

    # label_all = ['VDa-', 'Rg-', 'Vb-', 'Cc-', 'Ca-',
    #             'Nc-', 'Na-', 'Va-', 'VDb-', 'Vd-',
    #             'VDc-', 'Ra-', 'Rb-', 'Vc-', 'Nb-', 'Rf-', 'Rh-']
    label_all_dic = {}
    i = 0
    for jf in tqdm(json_files):
        json_path = os.path.join(json_dir, jf)
        with open(json_path, 'r') as f:
            info = json.load(f)

            roi_list = info["shapes"]
            print(roi_list)
            for roi in roi_list:
                label = roi["label"]
                if label not in label_all_dic:
                    label_all_dic[label] = i
                    i += 1
                else:
                    continue
    print(label_all_dic)


def json2txt(json_path, txt_path, label_all_dic):
    """
        将原始json数据集，转为txt格式
    """
    roi_list = []
    with open(json_path, 'r') as f:
        # 开始写txt
        label_txt = open(txt_path, "w")
        # 解析json
        info = json.load(f)
        roi_list = info["shapes"]
        h = info["imageHeight"]
        w = info["imageWidth"]

        for roi in roi_list:
            label = roi["label"]
            if label in label_all_dic:
                cls_id = label_all_dic[label]
            # 读json中的roi点， 左下点，右上点
            point = roi["points"]
            x1, y1 = point[0]
            x2, y2 = point[1]
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            b = (float(x_min), float(x_max), float(y_min), float(y_max))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            label_txt.write(str(cls_id) + " " +
                            " ".join([str(a) for a in bb]) + '\n')

        label_txt.close()


def txt2yolo_main():
    """
        遍历文件夹，将txt label转为yolo
    """
    # target_img_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/images/train"
    # t_label_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/labels/train_txt_label"
    # yolo_label_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/labels/train"

    target_img_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/images/val"
    t_label_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/labels/val_real"
    yolo_label_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/labels/val"

    patch_dirs = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset_ext/base_nonoise_newcls"
    label_list = os.listdir(patch_dirs)

    label_all_dic = {}
    for i in range(len(label_list)):
        label_all_dic[label_list[i]] = i
    print('label cls:', label_all_dic)

    t_label_list = os.listdir(t_label_dir)
    if not os.path.exists(yolo_label_dir):
        os.mkdir(yolo_label_dir)

    for t_l in tqdm(t_label_list):
        t_label_path = os.path.join(t_label_dir, t_l)
        yolo_label_path = os.path.join(yolo_label_dir, t_l)
        x = t_l.split('.')
        img_name, ext = x
        t_img_path = os.path.join(target_img_dir, img_name+'.jpg')
        print(t_label_path)
        print(t_img_path)
        print("=====")
        t_img = cv2.imread(t_img_path)

        #
        txt2yolo(t_label_path, yolo_label_path, label_all_dic, t_img)


def txt2yolo(t_label_path, yolo_label_path, label_all_dic, img):
    """
        将txtlabel转为yolo
        txt每行： "label x1 y1 x2 y2"  （类别标签 左上角坐标 右上角坐标)
        yolo： label_id x_center y_center pw ph
    """
    h, w, _ = img.shape
    with open(t_label_path, 'r') as f:
        yolo_label_txt = open(yolo_label_path, "w")
        # r = []
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

            # ll =(x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]))
            # r.append(ll)
            label = x[0]

            if label in label_all_dic:
                cls_id = label_all_dic[label]

            x_min = min(float(x[1]), float(x[3]))
            x_max = max(float(x[1]), float(x[3]))
            y_min = min(float(x[2]), float(x[4]))
            y_max = max(float(x[2]), float(x[4]))

            b = (x_min, x_max, y_min, y_max)
            bb = convert((w, h), b)

            # print("cls:",cls)
            yolo_label_txt.write(str(cls_id) + " " +
                                 " ".join([str(a) for a in bb]) + '\n')

    f.close()


if __name__ == "__main__":
    # json_parse_main()
    txt2yolo_main()
