import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import json
# from psutil import OSX
from tqdm import tqdm
from parse_objfile import txt_parse

def xywh2xyxy_batch(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


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

def box_label(im, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    for i in range(len(box)):
        box[i] = int(box[i])
    pt1 = (box[0],box[1])
    pt2 = (box[2],box[3])

    cv2.rectangle(im, pt1, pt2,color=(0,255,255),thickness=3,lineType=0)
    cv2.putText(im, label,(box[0],box[1]+20),cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0,255,255),thickness=1,lineType=0)
    # cv2.imshow('im',im)
    # cv2.waitKey(0) 
    return im


def box_label_yl5(im, box, label='', color=(0, 128, 128), txt_color=(255, 255, 255),line_width=3):
    """
        yolov5中的标框
        box 为 xyxy
    """
    lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return im


def json_parse(json_path):
    with open(json_path, 'r') as f:
        # 解析json
        info = json.load(f)
        roi_list = info["shapes"]
    return roi_list


def draw_one_label(img_path, label_path,save_dir,cls=None):
    img = cv2.imread(img_path)

    # roi_list = json_parse(label_path)
    roi_list = txt_parse(label_path)

    for roi in roi_list:
        # json
        # label = roi["label"]
        # points = roi["points"]
        # x1,y1= points[0]
        # x2,y2 = points[1]

        # txt
        if cls is None:
            label = roi[0]
            x1,y1,x2,y2 = roi[1:]
            box = [x1,y1,x2,y2] 
            img = box_label(img, box, label=label)
        else:  # yolo xywh format
            label = roi[0]
            label = cls[int(label)]
            box = list(roi[1:])
            # temp_box = np.array(box).reshape(1, 4)
            # gn = np.array(img.shape)[[1,0,1,0]]
            # temp_box = (temp_box * gn).reshape(-1)
            # box = temp_box.tolist()
            
            # box = xywh2xyxy(box) 

            img = box_label(img, box, label)

    # cv2.imshow('im',img)
    # cv2.waitKey(0)

    file_name = os.path.basename(img_path)
    file_path = os.path.join(save_dir,file_name)
    # print(file_path)
    cv2.imwrite(file_path, img)

def draw_label_main():
    label_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_labeled"
    image_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_imgs"
    save_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_gt_imgs"
    os.makedirs(save_dir,exist_ok=True)

    # 
    # classes = ["capacitor","resistor", "transistor",
    #             "ic", "pad", "inductor","others"]
    classes = ["component"]
    image_list = os.listdir(image_dir)
    for img_path in tqdm(image_list):
        filename = os.path.basename(img_path).split(".")[0]
        label_path = os.path.join(label_dir, filename+'.txt')

        img_path = os.path.join(image_dir,img_path)
        # draw_one_label(img_path, label_path,save_dir, cls=classes)
        draw_one_label(img_path, label_path,save_dir)

# def draw_label_main2():
#     label_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_labeled"
#     image_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white"
#     save_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_gt_img"
#     os.makedirs(save_dir,exist_ok=True)

#     for i in range(13):
#         print(i)
#         img_sub_dir = os.path.join(image_dir, str(i))
#         file_list = os.listdir(img_sub_dir)
#         for file in file_list:
#             os.path.join(img_sub_dir, file)


#         classes = ["component"]
#         image_list = os.listdir(image_dir)
#         for img_path in tqdm(image_list):
#             filename = os.path.basename(img_path).split(".")[0]
#             label_path = os.path.join(label_dir, filename+'.txt')

#             img_path = os.path.join(image_dir,img_path)
#             draw_one_label(img_path, label_path,save_dir, cls=classes)


def draw_crop_line(base_images_dir, outdir, i=None, sliceHeight=1024, sliceWidth=1024,
             zero_frac_thresh=0.2, overlap=0.2, verbose=True):
    """
    画出切图时候 的切线.
    """
    cnt = 0
    base_img_list = os.listdir(base_images_dir)

    for per_img_name in tqdm(base_img_list):

        out_name, _ = os.path.splitext(per_img_name)
        image_path = os.path.join(base_images_dir, per_img_name)

        image0 = cv2.imread(image_path, 1)  # color
        ext = '.' + image_path.split('.')[-1]

        win_h, win_w = image0.shape[:2]

        pad = 0
        if sliceHeight > win_h:
            pad = sliceHeight - win_h
        if sliceWidth > win_w:
            pad = max(pad, sliceWidth - win_w)
        # pad the edge of the image with black pixels
        if pad > 0:
            border_color = (0, 0, 0)
            image0 = cv2.copyMakeBorder(image0, pad, pad, pad, pad,
                                        cv2.BORDER_CONSTANT, value=border_color)
        win_size = sliceHeight * sliceWidth

        n_ims = 0
        n_ims_nonull = 0
        dx = int((1. - overlap) * sliceWidth)   # 153
        dy = int((1. - overlap) * sliceHeight)

        imgout = image0.copy()

        for y0 in range(0, image0.shape[0], dy):
            for x0 in range(0, image0.shape[1], dx):
                n_ims += 1
                if y0 + sliceHeight > image0.shape[0]:
                    y = image0.shape[0] - sliceHeight
                else:
                    y = y0
                if x0 + sliceWidth > image0.shape[1]:
                    x = image0.shape[1] - sliceWidth
                else:
                    x = x0

                # window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
                box = [x, y, x+sliceWidth,y+sliceHeight]
                imgout = box_label(imgout, box)

        cv2.imwrite("x.png", imgout)


def zooming(img, scale=0.1):
    img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    return img

def draw_crop_line_test():
    base_images_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/ori_img"
    outdir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/ddd"
    draw_crop_line(base_images_dir, outdir, i=None, sliceHeight=1024, sliceWidth=1024,
             zero_frac_thresh=0.2, overlap=0.2, verbose=True)


if __name__ == "__main__":
    draw_label_main()
    # draw_one_label()
    # draw_crop_line_test()