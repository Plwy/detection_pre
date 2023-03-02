from distutils.cygwinccompiler import Mingw32CCompiler
from email.mime import base
import os
from tkinter import W, Y
import cv2
import time
import codecs
import xml.etree.ElementTree as ET
from matplotlib.pyplot import box
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

"""
输入一张大图
先将大图按照滑动窗口 分割成小图. 存储
输入 检测模型得到检测结果,图像和 txt
将所有结果回归到大图, 采用nms去除多余框
得到最终的结果
"""
classes = ["component"]
def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def deal_xml(xml_f):
    tree = ET.parse(xml_f)
    root = tree.getroot()
    object_list=[]
    # 处理每个标注的检测框
    for obj in get(root, 'object'):
        # 取出检测框类别名称
        category = get_and_check(obj, 'name', 1).text
        # 更新类别ID字典
        bndbox = get_and_check(obj, 'bndbox', 1)
        xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
        ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
        xmax = int(get_and_check(bndbox, 'xmax', 1).text)
        ymax = int(get_and_check(bndbox, 'ymax', 1).text)
        assert (xmax > xmin)
        assert (ymax > ymin)
        o_width = abs(xmax - xmin)
        o_height = abs(ymax - ymin)
        obj_info=[xmin,ymin,xmax,ymax,category]
        object_list.append(obj_info)
    return object_list


def compute_IR(rec1,rec2):
    """
    计算两个矩形框的交集占rec1的比例
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交集比例Intersection Ratio
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
        return S_cross/S1


def exist_objs(slice_box,all_objs_list, sliceHeight, sliceWidth):
    '''
    slice_box:当前slice的图像边框
    all_objs_list:原图中的所有目标
    return:原图中位于当前slice中的目标集合
    '''
    return_objs=[]
    min_h, min_w = 20, 20 #35, 35  # 有些目标GT会被窗口切分，太小的丢掉
    s_xmin, s_ymin, s_xmax, s_ymax = slice_box[0], slice_box[1], slice_box[2], slice_box[3]
    for vv in all_objs_list:
        category, xmin, ymin, xmax, ymax=vv[0],vv[1],vv[2],vv[3],vv[4]
        
        rec1 = vv[1:]
        rec2 = slice_box
        IR = compute_IR(rec1,rec2)
        if IR < 0.4:
            continue
        else:
            # 1111111
            if s_xmin<=xmin<s_xmax and s_ymin<=ymin<s_ymax:  # 目标点的左上角在切图区域中
                if s_xmin<xmax<=s_xmax and s_ymin<ymax<=s_ymax:  # 目标点的右下角在切图区域中
                    x_new=xmin-s_xmin
                    y_new=ymin-s_ymin
                    return_objs.append([x_new,y_new,x_new+(xmax-xmin),y_new+(ymax-ymin),category])
            if s_xmin<=xmin<s_xmax and ymin < s_ymin:  # 目标点的左上角在切图区域上方
                # 22222222
                if s_xmin < xmax <= s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域中
                    x_new = xmin - s_xmin
                    y_new = 0
                    if xmax - s_xmin - x_new > min_w and ymax - s_ymin - y_new > min_h:
                        return_objs.append([x_new, y_new, xmax - s_xmin, ymax - s_ymin, category])

                # 33333333
                if xmax > s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域右方
                    x_new = xmin - s_xmin
                    y_new = 0
                    if s_xmax-s_xmin - x_new > min_w and ymax - s_ymin - y_new > min_h:
                        return_objs.append([x_new, y_new, s_xmax-s_xmin, ymax - s_ymin, category])
            if s_ymin < ymin <= s_ymax and xmin < s_xmin:  # 目标点的左上角在切图区域左方
                # 444444
                if s_xmin < xmax <= s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域中
                    x_new = 0
                    y_new = ymin - s_ymin
                    if xmax - s_xmin - x_new > min_w and ymax - s_ymin - y_new > min_h:
                        return_objs.append([x_new, y_new, xmax - s_xmin, ymax - s_ymin, category])
                # 555555
                if s_xmin < xmax < s_xmax and ymax >= s_ymax:   # 目标点的右下角在切图区域下方
                    x_new = 0
                    y_new = ymin - s_ymin
                    if xmax - s_xmin - x_new > min_w and s_ymax - s_ymin - y_new > min_h:
                        return_objs.append([x_new, y_new, xmax - s_xmin, s_ymax - s_ymin, category])
            # 666666
            if s_xmin >= xmin  and ymin <= s_ymin:  # 目标点的左上角在切图区域左上方
                if s_xmin<xmax<=s_xmax and s_ymin<ymax<=s_ymax:  # 目标点的右下角在切图区域中
                    x_new = 0
                    y_new = 0
                    if xmax - s_xmin - x_new > min_w and ymax - s_ymin - y_new > min_h:
                        return_objs.append([x_new, y_new, xmax - s_xmin, ymax - s_ymin, category])
            # 777777
            if s_xmin <= xmin < s_xmax and s_ymin <= ymin < s_ymax:  # 目标点的左上角在切图区域中
                if ymax >= s_ymax and xmax >= s_xmax:              # 目标点的右下角在切图区域右下方
                    x_new = xmin - s_xmin
                    y_new = ymin - s_ymin
                    if s_xmax - s_xmin - x_new > min_w and s_ymax - s_ymin - y_new > min_h:
                        return_objs.append([x_new, y_new, s_xmax - s_xmin, s_ymax - s_ymin, category])
                # 8888888
                if s_xmin < xmax < s_xmax and ymax >= s_ymax:  # 目标点的右下角在切图区域下方
                    x_new = xmin - s_xmin
                    y_new = ymin - s_ymin
                    if xmax - s_xmin - x_new > min_w and s_ymax - s_ymin - y_new > min_h:
                        return_objs.append([x_new, y_new, xmax - s_xmin, s_ymax - s_ymin, category])
                # 999999
                if xmax > s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域右方
                    x_new = xmin - s_xmin
                    y_new = ymin - s_ymin
                    if s_xmax - s_xmin - x_new > min_w and ymax - s_ymin - y_new > min_h:
                        return_objs.append([x_new, y_new, s_xmax - s_xmin, ymax - s_ymin, category])

    return return_objs


def make_slice_voc(out_anno_path, exiset_obj_list,sliceHeight=1024, sliceWidth=1024):
    file_name, ext = os.path.splitext(os.path.basename(out_anno_path))
    #
    print("====",out_anno_path)
    with codecs.open(out_anno_path, 'w', 'utf-8') as xml:
        xml.write('<annotation>\n')
        xml.write('\t<filename>' + file_name + '</filename>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(sliceWidth) + '</width>\n')
        xml.write('\t\t<height>' + str(sliceHeight) + '</height>\n')
        xml.write('\t\t<depth>' + str(3) + '</depth>\n')
        xml.write('\t</size>\n')
        cnt = 1
        for obj in exiset_obj_list:
            #
            bbox = obj[:4]
            class_name = obj[-1]
            if not isinstance(class_name, str):
                class_name = classes[int(class_name)]
            xmin, ymin, xmax, ymax = bbox
            #
            xml.write('\t<object>\n')
            xml.write('\t\t<name>' + class_name + '</name>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')
            cnt += 1
        assert cnt > 0
        xml.write('</annotation>')


def make_slice_txt(txt_path, obj_list):
    """
    [x,y,x,y,conf,c] or [x,y,x,y,c]
    output txt:
    [c, x,y,x,y]or [c, x,y,x,y,conf]
    """
    label_txt = open(txt_path, "w")
    for obj in obj_list:
        if len(obj)==5:
            classid = str(obj[-1])
            label_txt.write(classid + " " + " ".join([str(a) for a in obj[:4]]) + '\n')
        elif len(obj)==6:
            classid = str(int(obj[-1]))
            conf = str(obj[-2])
            label_txt.write(classid + " " + " ".join([str(a) for a in obj[:4]]) + " " + conf + '\n')     
    label_txt.close()


def slice_im_anno(base_images_dir, base_anno_dir, outdir, out_anno_dir,sliceHeight=640, sliceWidth=640,
             zero_frac_thresh=0.2, overlap=0.2, verbose=True):
    """
        从上到下从左到右进行切, 
        overlap:相邻两图的重叠度设置
        ##保留没有标注目标的空图
    """
    cnt = 0
    emp_cnt = 0  # 记录切出来的没有元件的空图
    base_img_list = os.listdir(base_images_dir)

    for per_img_name in tqdm(base_img_list):
        out_name, _ = os.path.splitext(per_img_name)
        image_path = os.path.join(base_images_dir, per_img_name)
        ann_path = os.path.join(base_anno_dir, per_img_name[:-4] + '.xml')

        image0 = cv2.imread(image_path, 1)  # color
        ext = '.' + image_path.split('.')[-1]
        win_h, win_w = image0.shape[:2]

        # if slice sizes are large than image, pad the edges
        # 避免出现切图的大小比原图还大的情况
        object_list = deal_xml(ann_path)
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

        t0 = time.time()
        n_ims = 0    # 切图编号
        n_ims_nonull = 0
        dx = int((1. - overlap) * sliceWidth)   # 153
        dy = int((1. - overlap) * sliceHeight)

        for y0 in range(0, image0.shape[0], dy):
            for x0 in range(0, image0.shape[1], dx):                     
                n_ims += 1
                #
                #这一步确保了不会出现比要切的图像小的图，其实是通过调整最后的overlop来实现的
                #举例:h=6000,w=8192.若使用640来切图,overlop:0.2*640=128,间隔就为512.所以小图的左上角坐标的纵坐标y0依次为:
                #:0,512,1024,....,5120,接下来并非为5632,因为5632+640>6000,所以y0=6000-640
                if y0 + sliceHeight > image0.shape[0]:
                    y = image0.shape[0] - sliceHeight
                else:
                    y = y0
                if x0 + sliceWidth > image0.shape[1]:
                    x = image0.shape[1] - sliceWidth
                else:
                    x = x0

                slice_xmax = x + sliceWidth
                slice_ymax = y + sliceHeight

                window_c = image0[y:y + sliceHeight, x:x + sliceWidth]

                crop_name = out_name + '_crop' + str(cnt) + \
                                    '|' + str(x) + '_' + str(y) + '_' + str(sliceWidth) + '_' + str(sliceHeight) + \
                                    '_' + str(pad)
                outpath = os.path.join(outdir,  crop_name + ext)
                cv2.imwrite(outpath, window_c)
                cnt += 1
                exiset_obj_list = exist_objs([x, y, slice_xmax, slice_ymax], 
                                                object_list, sliceHeight, sliceWidth)
                
                # 如果为空,说明切出来的这一张图不存在(符合要求的)目标
                if exiset_obj_list!=[]:  
                    # extract image
                    window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
                    # get black and white image
                    window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)

                    # find threshold that's not black
                    # skip if image is mostly empty
                    ret, thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
                    non_zero_counts = cv2.countNonZero(thresh1)
                    zero_counts = win_size - non_zero_counts
                    zero_frac = float(zero_counts) / win_size
                    if zero_frac >= zero_frac_thresh:
                        if verbose:
                            print("Zero frac too high at:", zero_frac)
                        continue
                    else:
                        out_anno_path = os.path.join(out_anno_dir, crop_name+'.xml')
                        make_slice_voc(out_anno_path, exiset_obj_list, sliceHeight, sliceWidth)


def slice_im(base_images_dir, outdir, base_anno_dir=None, out_anno_dir=None,sliceHeight=640, sliceWidth=640,
             zero_frac_thresh=0.2, overlap=0.2, verbose=True):
    """
        从上到下从左到右进行切, overlap:相邻两图的重叠度设置
        ##保留没有标注目标的空图
    """
    cnt = 0
    # base_img_list = os.listdir(base_images_dir)

    # for per_img_name in tqdm(base_img_list):
    for cur, sub_dir, files in os.walk(base_images_dir):
        if len(files) == 0:
            continue
        else:
            for img_name in tqdm(files):
                out_name, _ = os.path.splitext(img_name)
                outsub_dir = os.path.join(outdir,cur.split('/')[-1])
                os.makedirs(outsub_dir,exist_ok=True)
                outcrop_dir = os.path.join(outsub_dir, out_name)
                # 子目录存放 
                if not os.path.exists(outcrop_dir):
                    os.mkdir(outcrop_dir)

                image_path = os.path.join(cur, img_name)

                image0 = cv2.imread(image_path, 1)  # color
                ext = '.' + image_path.split('.')[-1]
                win_h, win_w = image0.shape[:2]

                # 避免出现切图的大小比原图还大的情况
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

                t0 = time.time()
                n_ims = 0    # 切图编号
                n_ims_nonull = 0
                dx = int((1. - overlap) * sliceWidth)   # 153
                dy = int((1. - overlap) * sliceHeight)

                object_list =[]

                # 被切大图带标签
                if base_anno_dir is not None:
                    ann_path = os.path.join(base_anno_dir, img_name[:-4] + '.xml')
                    if os.path.isfile(ann_path):   
                        object_list = deal_xml(ann_path)

                for y0 in range(0, image0.shape[0], dy):
                    for x0 in range(0, image0.shape[1], dx):                     
                        n_ims += 1
                        #
                        #这一步确保了不会出现比要切的图像小的图，其实是通过调整最后的overlop来实现的
                        #举例:h=6000,w=8192.若使用640来切图,overlop:0.2*640=128,间隔就为512.所以小图的左上角坐标的纵坐标y0依次为:
                        #:0,512,1024,....,5120,接下来并非为5632,因为5632+640>6000,所以y0=6000-640
                        if y0 + sliceHeight > image0.shape[0]:
                            y = image0.shape[0] - sliceHeight
                        else:
                            y = y0
                        if x0 + sliceWidth > image0.shape[1]:
                            x = image0.shape[1] - sliceWidth
                        else:
                            x = x0

                        slice_xmax = x + sliceWidth
                        slice_ymax = y + sliceHeight

                        window_c = image0[y:y + sliceHeight, x:x + sliceWidth]

                        crop_name = out_name + '_crop' + str(cnt) + \
                                            '|' + str(x) + '_' + str(y) + '_' + str(sliceWidth) + '_' + str(sliceHeight) + \
                                            '_' + str(pad)
                        
                        outpath = os.path.join(outcrop_dir, crop_name + ext)
                        cv2.imwrite(outpath, window_c)
                        cnt += 1

                        
                        if len(object_list) == 0:   # 被切的大图不带标签
                            continue
                        else:
                            exiset_obj_list = exist_objs([x, y, slice_xmax, slice_ymax], 
                                                            object_list, sliceHeight, sliceWidth)
                                
                            # 如果为空,说明切出来的这一张图不存在(符合要求的)目标
                            if exiset_obj_list!=[]:  
                                # extract image
                                window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
                                # get black and white image
                                window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)

                                # find threshold that's not black
                                # skip if image is mostly empty
                                ret, thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
                                non_zero_counts = cv2.countNonZero(thresh1)
                                zero_counts = win_size - non_zero_counts
                                zero_frac = float(zero_counts) / win_size
                                # print "zero_frac", zero_fra
                                if zero_frac >= zero_frac_thresh:
                                    if verbose:
                                        print("Zero frac too high at:", zero_frac)
                                    continue
                                # else save
                                else:
                                    # outpath = os.path.join(outdir,  out_name + '_crop' + str(cnt) + ext)
                                    # cv2.imwrite(outpath, window_c)
                                    # n_ims_nonull += 1
                                    #------制作新的xml------
                                    out_anno_path = os.path.join(out_anno_dir, crop_name+'.xml')
                                    make_slice_voc(out_anno_path, exiset_obj_list, sliceHeight, sliceWidth)


def crop_imgs(base_images_dir, crops_dir,base_ann_dir=None,crop_anno_dir=None):
    """大图切小图
    """
    if not os.path.exists(crops_dir):
        os.makedirs(crops_dir)
    # 带标签的切图
    if base_ann_dir is not None:
        if not os.path.exists(crop_anno_dir):
            os.makedirs(crop_anno_dir)
        slice_im_anno(base_images_dir, base_ann_dir, crops_dir,crop_anno_dir, sliceHeight=1024, sliceWidth=1024)
    else:
        slice_im(base_images_dir,  crops_dir, sliceHeight=1024, sliceWidth=1024)


def crop_main():
    # base_images_dir = '/media/zsl/data/zsl_datasets/PCB_test/HR_test/2_img'   # 这里就是原始的图片
    # base_ann_dir = '/media/zsl/data/zsl_datasets/PCB_test/HR_test/2_anno'
    # crop_anno_dir = '/media/zsl/data/zsl_datasets/PCB_test/HR_test/2_crop_anno'  # 切出来的标签也保存为voc格式
    # crops_dir = '/media/zsl/data/zsl_datasets/PCB_test/HR_test/2_crop'
    base_images_dir = '/media/zsl/data/zsl_datasets/PCB_test/HR_test2/1_compare2/defects_aligned_whiteboard'   # 这里就是原始的图片
    crops_dir = '/media/zsl/data/zsl_datasets/PCB_test/HR_test2/1_compare2/defects_aligned_whiteboard_crop'
    base_ann_dir = None
    crop_anno_dir = None  # 切出来的标签也保存为voc格式

    crop_imgs(base_images_dir, crops_dir, base_ann_dir=base_ann_dir,crop_anno_dir=crop_anno_dir)


def stitch_main():
    """
        图片拼接
    """
    crops_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test/1_crop"
    ori_path = '/media/zsl/data/zsl_datasets/PCB_test/HR_test/1_img/template_3.jpg'
    stitch_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test/1_stitch"

    ori_img = cv2.imread(ori_path)
    img0 = np.zeros_like(ori_img)

    print(img0.shape)
    crop_paths = glob.glob(crops_dir+'/*')
    for crop_path in crop_paths:
        save_path = os.path.join(stitch_dir, os.path.basename(ori_path))

        crop = cv2.imread(crop_path)

        crop_name, ext = os.path.splitext(os.path.basename(crop_path))
        crop_mark_s = crop_name.split('|')[-1]
        crop_mark = crop_mark_s.split('_')
        print("==", crop_mark)
        y = int(crop_mark[0])
        x = int(crop_mark[1])
        sliceh = int(crop_mark[2])
        slicew = int(crop_mark[3])
        pad = int(crop_mark[4])
        print(crop.shape)
        if pad > 0:
            img0 = ori_img
        else:
            img0[y:y+sliceh, x:x+slicew, :] = crop

        cv2.imwrite(save_path, img0)

def txt_parse_conf(txt_path):
    """
        带置信度
    """
    r = []
    with open(txt_path, 'r') as f:  
        a = f.readlines()
        for l in a:
            l = l.strip()
            x = l.split(' ')
            ll =(float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[0]))
            r.append(ll)
        f.close()
    return r


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

def box_label_yl5(im, box, label='', color=(0, 255, 255), txt_color=(255, 255, 255),line_width=3):
    """
        yolov5中的标框
    """
    lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
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



def cluster_merge(cluster):
    """
        将列表内的相关元素进行聚合, 直接关联和间接关联元素整合到一起.
        eg:
        input:[[0],[1,3],[2],[3,0],[4,2]]
        result:[[0,1,3],[2,4]]
    """
    # cluster = [[0],[1,3,5],[2,8],[3,1,4],[4],[5,3],[6],[7],[8,2]]
    flag = [False]*len(cluster)
    cluster2 = []

    for i in range(len(cluster)):
        if flag[i]:
            continue
        else:
            flag[i]=True

        p = cluster[i]
        if len(p)==1:
            cluster2.append(p)
            continue
        else:
            temp=p  
            for j in p:
                if flag[j]:
                    continue
                else:
                    flag[j] = True
                    if len(cluster[j]) == 1:
                        continue
                    else:
                        temp.extend(cluster[j])
            temp = list(set(temp))
            cluster2.append(temp)
                    
    return cluster2


def merge_obj(cluster_obj_indexs, all_obj_list, conf_thred = 0.6):
    """聚合 相关矩形. 取最大外接矩形. 目前只考虑单类
    """
    cluster_obj = []
    for indexs in cluster_obj_indexs:
        cx_min = 100000
        cy_min = 100000
        cx_max = 0
        cy_max = 0
        c_conf = 0   # 取聚合里所有框最大的置信度

        for index in indexs :
            obj = all_obj_list[index]
            x_min, y_min, x_max, y_max, conf, c = obj
            if x_min < cx_min:
                cx_min = x_min
            if y_min < cy_min:
                cy_min = y_min
            if x_max > cx_max:
                cx_max = x_max
            if y_max > cy_max:
                cy_max = y_max
            if conf > c_conf:
                c_conf = conf

        if c_conf < conf_thred:
            continue
        else:
            cluster_obj.append([cx_min,cy_min,cx_max,cy_max,c_conf,c])

    return cluster_obj

def zooming(img, scale=0.1):
    img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    return img

def is_boundry(box, slice=1024, overlap=0.2, thred = 10):
    flag = False

    inarea = slice*overlap
    outarea = slice*(1-overlap)
    # print(inarea, outarea)
    for x in box:
        a, b= x//outarea, x%outarea
        if 0 <= b <= thred or b >= outarea-thred:
            flag = True
            break
        
        elif a > 0 and inarea-thred <= b <= inarea+thred:
            flag = True
            break
        else:
            continue

    return  flag



def get_puzzle_box(all_obj_list,slicew=1024,sliceh=1024,overlap=0.2):
    # 先粗暴遍历
    cluster_obj = []   # 初步的聚合结果
    for i, obj0 in enumerate(all_obj_list):
        flag = False   # box 是否为切边的框
        x0_min, y0_min, x0_max, y0_max, conf0, c0 = obj0
        flag = is_boundry(obj0[:4])

        if flag:
            cluster_obj.append([x0_min, y0_min, x0_max, y0_max, conf0, c0])
      
    return cluster_obj

        # for j, obj1 in enumerate(all_obj_list):
        #     x1_min, y1_min, x1_max, y1_max, conf1, c1 = obj1
                
        #     # 2.获取两个框的交集框,面积占比
        #     # - 1.交集几乎占某个box的全部 2.完全包含关系  3. 交集区域大判定为大目标 4. 交集属于overlap
        #     cross_x_min = max(x0_min, x1_min)
        #     cross_x_max = min(x0_max, x1_max)
        #     cross_y_min = max(y0_min, y1_min)
        #     cross_y_max = min(y0_max, y1_max)
        #     # # ## 判定 交集框的位置
        #     overlap_w = slicew*overlap
        #     in_w = slicew*(1-overlap)
        #     overlap_h = sliceh*overlap
        #     in_h = sliceh*(1-overlap)


def box_check(ori_img, all_obj_list, slicew=1024, sliceh=1024, overlap=0.2, large_size=20):
    """
        得到所有切图的检测框后,针对大目标的分离框进行初步聚合.
        当前设定暂在717数据上有效.旋转图上无效
    """
    c_h_min = 1000000
    c_w_min = 1000000
    c_h_max = 0
    c_w_max = 0
    for obj in all_obj_list:
        x_min, y_min, x_max, y_max, conf, c = obj
        c_h,c_w = y_max - y_min, x_max - x_min
        if c_h < c_h_min:c_h_min = c_h
        if c_w < c_w_min:c_w_min = c_w
        if c_h > c_h_max:c_h_max = c_h
        if c_w > c_w_max:c_w_max = c_w    
    print(c_w_min, c_h_min, c_w_max, c_h_max)

    # 先粗暴遍历
    cluster = []   # 初步的聚合结果
    for i, obj0 in enumerate(all_obj_list):
        sub = []
        x0_min, y0_min, x0_max, y0_max, conf0, c0 = obj0
        for j, obj1 in enumerate(all_obj_list):
            x1_min, y1_min, x1_max, y1_max, conf1, c1 = obj1

            # 2.获取两个框的交集框,面积占比
            # - 1.交集几乎占某个box的全部 2.完全包含关系  3. 交集区域大判定为大目标 4. 交集属于overlap
            cross_x_min = max(x0_min, x1_min)
            cross_x_max = min(x0_max, x1_max)
            cross_y_min = max(y0_min, y1_min)
            cross_y_max = min(y0_max, y1_max)

            #两矩形无相交区域的情况
            if cross_x_min >= cross_x_max or cross_y_max <= cross_y_min:
                continue
            # 有交集
            S1 = (x0_max-x0_min)*(y0_max-y0_min)
            S2 = (x1_max-x1_min)*(y1_max-y1_min)
            S_cross = (cross_x_max-cross_x_min)*(cross_y_max-cross_y_min)  
            # 交集面积占某个目标的一半多, 认定为一个目标进行合并
            if S_cross/S1 > 0.9 or S_cross/S2 > 0.9:
                ## 添加交集约束,防止小元件被大元件框并入
                sub.append(j)
            elif S_cross/S1 > 0.1 or S_cross/S2 > 0.1:
                # 两个框有一个边长几乎是相同的
                w0 = x0_max - x0_min
                w1 = x1_max - x1_min
                h0 = y0_max - y0_min
                h1 = y1_max - y1_min

                ratioh = min(h0,h1)/max(h0,h1)
                ratiow = min(w0,w1)/max(w0,w1)

                if ratiow > 0.8 or ratioh > 0.8:
                    sub.append(j)
                # if abs(w0-w1) < min(w0,w1)/7 or abs(h0-h1) < min(h0,h1)/7:
                #     sub.append(j)
                # 交集的长宽大小大于 大目标的长宽值, 判定为大目标
                if cross_x_max-cross_x_min > large_size or cross_y_max-cross_x_min > large_size:
                    # sub.append(j)
                    pass
                else:
                    continue
                
        cluster.append(sub)
    # print("merge1====",cluster,len(cluster))
    # ## 得到相关框初步的聚合结果
    cluster_final = cluster_merge(cluster)   # 357
    # print("merge2====",cluster_final, len(cluster_final)) #166
    return cluster_final


def box_check_2(ori_img, all_obj_list, slicew=1024, sliceh=1024, overlap=0.2, large_size=20):
    """
       用于修改合并逻辑的测试函数.
       添加切图边界约束;添加交集框长宽大小偏移大小占比约束
    """

    # 先粗暴遍历
    cluster = []   # 初步的聚合结果
    for i, obj0 in enumerate(all_obj_list):
        sub = []
        x0_min, y0_min, x0_max, y0_max, conf0, c0 = obj0
        
        for j, obj1 in enumerate(all_obj_list):
            if i == j:
                sub.append(j)
                continue

            x1_min, y1_min, x1_max, y1_max, conf1, c1 = obj1

            # 2.获取两个框的交集框,面积占比
            cross_x_min = max(x0_min, x1_min)
            cross_x_max = min(x0_max, x1_max)
            cross_y_min = max(y0_min, y1_min)
            cross_y_max = min(y0_max, y1_max)

            #两矩形无相交区域的情况
            if cross_x_min >= cross_x_max or cross_y_max <= cross_y_min:
                continue

            cross_box = [cross_x_min,cross_y_min,cross_x_max,cross_y_max]

            # 有交集
            S1 = (x0_max-x0_min)*(y0_max-y0_min)
            S2 = (x1_max-x1_min)*(y1_max-y1_min)
            S_cross = (cross_x_max-cross_x_min)*(cross_y_max-cross_y_min)  
            iou = S_cross/(S1+S2-S_cross)
            if iou > 0.5:
                sub.append(j)
            else:
                # 交集面积占某个目标的一半多, 认定为一个目标进行合并
                if S_cross/S1 > 0.9 or S_cross/S2 > 0.9:
                    sub.append(j)

                elif S_cross/S1 > 0.1 or S_cross/S2 > 0.1: 
                    if is_boundry(cross_box):  # 判定 交集框的位置, 如果交集框在切边上 其很大可能为被切分件
                        w0 = x0_max - x0_min
                        w1 = x1_max - x1_min
                        h0 = y0_max - y0_min
                        h1 = y1_max - y1_min
                        # 取左上角或右下角的最小偏移
                        offsetw = min(abs(x0_min - x1_min),abs(x0_max - x1_max))
                        offseth = min(abs(y0_min - y1_min), abs(y0_max - y1_max))

                        ratioh = min(h0,h1)/max(h0,h1)
                        ratiow = min(w0,w1)/max(w0,w1)
                        # if ratiow > 0.8 or ratioh > 0.8: # 两个框有一个边长几乎是相等
                        #     sub.append(j)
                        if ratiow > 0.8 or ratioh > 0.8: # 两个框有一个边长几乎是相等, 且错开不多
                            if offsetw < min(w0,w1)/10 or offseth < min(h0,h1)/10 :
                                sub.append(j)
                    else:
                        continue

        cluster.append(sub)
    # print("merge1====",cluster,len(cluster))
    # ## 得到相关框初步的聚合结果
    cluster_final = cluster_merge(cluster)   # 357
    # print("merge2====",cluster_final, len(cluster_final)) #166
    return cluster_final


def is_cropbox(box, slicex, slicey, slicew, sliceh, overlap=0.2):
    x_min, y_min, x_max, y_max = box
    # bw = x_max - x_min
    # bh = y_max - y_min

    aw = slicew*0.2*0.5
    mid_xl = slicex + aw
    mid_xr = (slicex + slicew) - aw
    ah = sliceh*0.2*0.5
    mid_yu = slicey + ah
    mid_yd = (slicey + sliceh) - ah

    flag = False
    if x_max < mid_xl or x_min > mid_xr or y_max < mid_yu or y_min > mid_yd:
        flag = True

    return flag




def stitch_test(crops_anno_dir, ori_path, stitch_dir, detect_dir, overlap=0.2):
    save_ext = '.txt' #'.xml'

    ori_img = cv2.imread(ori_path)
    img0 = np.zeros_like(ori_img)

    all_obj_list = []   # 所有切图的框
    crop_anno_num = 0

    anno_paths = glob.glob(crops_anno_dir+'/*')
    for anno_path in anno_paths:
        crop_name, ext = os.path.splitext(os.path.basename(anno_path))
        if not os.path.isfile(anno_path):
            continue

        else:
            # 得到当前切图的所有检测框
            # crop_object_list = deal_xml(anno_path)
            crop_object_list = txt_parse_conf(anno_path)
            crop_anno_num += 1
            # save_path = os.path.join(stitch_dir, os.path.basename(ori_path))
            # crop = cv2.imread(crop_path)
            if crop_name.find('|') < 0:    # 只取整图检测中的大件, 整图的小件误检高
                minsize = 100000
                maxsize = 0
                for obj in crop_object_list:
                    x_min, y_min, x_max, y_max, conf, c = obj
                    bw = x_max - x_min
                    bh = y_max - y_min
                    s = max(bw, bh)
                    if s <= minsize:
                        minsize = s
                    if s >= maxsize:
                        maxsize = s

                midsize = (maxsize+minsize)/2
                for obj in crop_object_list:
                    x_min, y_min, x_max, y_max, conf, c = obj
                    bw = x_max - x_min
                    bh = y_max - y_min
                    s = max(bw, bh)
                    if s > midsize:
                        all_obj_list.append([x_min, y_min, x_max, y_max, conf, c])
                    else:
                        continue
            else:

                crop_anno_num += 1
                # save_path = os.path.join(stitch_dir, os.path.basename(ori_path))
                # crop = cv2.imread(crop_path)
                crop_mark_s = crop_name.split('|')[-1]
                crop_mark = crop_mark_s.split('_')
                x = int(crop_mark[0])    
                y = int(crop_mark[1])
                slicew = int(crop_mark[2])
                sliceh = int(crop_mark[3])
                pad = int(crop_mark[4])

                for obj in crop_object_list:
                    x_min, y_min, x_max, y_max, conf, c = obj
                    new_x_min = x_min + x
                    new_y_min = y_min + y
                    new_x_max = x_max + x
                    new_y_max = y_max + y 
                    # # 框完全在切线中线的某一边就删掉.认为其可能带来误检
                    # if is_cropbox(obj[:4], x, y, slicew, sliceh, overlap):
                    #     continue
                    # else:
                    #     all_obj_list.append([new_x_min, new_y_min, new_x_max, new_y_max, conf, c])
                    all_obj_list.append([new_x_min, new_y_min, new_x_max, new_y_max, conf, c])

    print("all obj len:",len(all_obj_list))
    print("crop anno num:", crop_anno_num)
    # 处理大件物体
    # cluster_objs_indexs = box_check(ori_img, all_obj_list, overlap=0.2) # 相关性聚合, 返回聚合后的indexs
    cluster_objs_indexs = box_check_2(ori_img, all_obj_list, overlap=0.2) # 相关性聚合, 返回聚合后的indexs
    print("$$$$$$$$$$$",cluster_objs_indexs)
    cluster_objs = merge_obj(cluster_objs_indexs, all_obj_list, conf_thred=0.4)   # 返回筛选出的obj

    # cluster_objs = get_puzzle_box(all_obj_list)

    img = ori_img.copy()
    for obj in all_obj_list:
        box = obj[:4]
        conf = str(obj[4])
        img = box_label_yl5(img, box, label=conf)
        # img = box_label_yl5(img, box)

    img1 = ori_img.copy()
    for obj in cluster_objs:
        box = obj[:4]
        conf = str(obj[4])
        img1 = box_label_yl5(img1, box, label=conf)
        # img1 = box_label_yl5(img1, box)

    img_name = os.path.basename(ori_path)
    ori_annoimg_path = os.path.join(detect_dir, img_name.replace('.','_ori.'))
    merge_annoimg_path = os.path.join(detect_dir, img_name.replace('.','_merge.'))
    print("====",ori_annoimg_path)
    print("====",merge_annoimg_path)

    cv2.imwrite(ori_annoimg_path, img)
    cv2.imwrite(merge_annoimg_path, img1)


    ## save  labels  as xml
    stitch_anno_name = os.path.basename(ori_path).split('.')[0] + save_ext
    save_path = os.path.join(stitch_dir, stitch_anno_name)
    if save_ext=='.xml':
        make_slice_voc(save_path, cluster_objs, sliceHeight=1024, sliceWidth=1024)
    else:
        make_slice_txt(save_path, cluster_objs)
    ## save_labels as txt
    print("====",save_path)

def stitch_anno_main():
    """
    小图检测结果 拼接成大图
        ori_path 原始大图
        crops_dir 切好的小图,按规则命名
        crops_anno_dir  小图得到的检测结果框
        stitch_dir 拼接后的结果输出路径
    """
    crops_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/crops/多件_TEST38_aligned"
    crops_anno_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/anno/多件_TEST38_aligned/labels"
    ori_path = '/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/ori_img/x.png'
    save_dir= "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/anno_stitch1"
    detect_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/img_stitch1"

    # # 
    # crops_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/crops/多件_TEST38_aligned"
    # crops_anno_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/anno/多件_TEST38_aligned/labels"
    # ori_path = '/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/ori_img/x.png'
    # save_dir= "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/anno_stitch1"
    # detect_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/for_merge/img_stitch1"
    save_ext = '.txt' #'.xml'

    name, ext = os.path.splitext(os.path.basename(ori_path))
    stitch_dir = os.path.join(save_dir, name)
    os.makedirs(stitch_dir, exist_ok=True)
    os.makedirs(detect_dir, exist_ok=True)

    ori_img = cv2.imread(ori_path)
    img0 = np.zeros_like(ori_img)

    all_obj_list = []   # 所有切图的框
    crop_dic = []   # {'crop_name':obj_list}
    crop_paths = glob.glob(crops_dir+'/*')
    crop_anno_num = 0
    print(len(crop_paths))
    for crop_path in crop_paths:
        crop_name, ext = os.path.splitext(os.path.basename(crop_path))
        anno_path = os.path.join(crops_anno_dir, crop_name+'.txt')
        # print(anno_path)
        if not os.path.isfile(anno_path):
            continue
        else:
            #  带切图标记需要还原框位置
            if crop_name.find('|') > 0:
                crop_anno_num += 1
                # save_path = os.path.join(stitch_dir, os.path.basename(ori_path))
                # crop = cv2.imread(crop_path)
                crop_mark_s = crop_name.split('|')[-1]
                crop_mark = crop_mark_s.split('_')
                x = int(crop_mark[0])    
                y = int(crop_mark[1])
                slicew = int(crop_mark[2])
                sliceh = int(crop_mark[3])
                pad = int(crop_mark[4])

                # 得到当前切图的所有检测框
                # crop_object_list = deal_xml(anno_path)
                crop_object_list = txt_parse_conf(anno_path)

                for obj in crop_object_list:
                    x_min, y_min, x_max, y_max, conf, c = obj
                    new_x_min = x_min + x
                    new_y_min = y_min + y
                    new_x_max = x_max + x
                    new_y_max = y_max + y 
                    all_obj_list.append([new_x_min, new_y_min, new_x_max, new_y_max, conf, c])
            else:
                crop_object_list = txt_parse_conf(anno_path)
                for obj in crop_object_list:
                    x_min, y_min, x_max, y_max, conf, c = obj
                    all_obj_list.append([new_x_min, new_y_min, new_x_max, new_y_max, conf, c])


    print("all obj len:",len(all_obj_list))
    print("crop anno num:", crop_anno_num)
    # # 处理大件物体
    cluster_objs_indexs = box_check_2(ori_img, all_obj_list, overlap=0.2) # 相关性聚合, 返回聚合后的indexs
    cluster_objs = merge_obj(cluster_objs_indexs, all_obj_list)   # 返回筛选出的obj

    # # 找到所有 边上的框
    # cluster_objs = get_puzzle_box(all_obj_list, overlap=0.2)
    # print(len(cluster_objs))

    img = ori_img.copy()
    for obj in all_obj_list:
        box = obj[:4]
        img = box_label_yl5(img, box, label="comp")

        # print("box:", box)
        # x = zooming(img, 0.2)
        # cv2.imshow('dd',x)
        # cv2.waitKey(0)

    img1 = ori_img.copy()
    for obj in cluster_objs:
        box = obj[:4]
        img1 = box_label_yl5(img1, box, label="comp")

    img_name = os.path.basename(ori_path)
    ori_annoimg_path = os.path.join(detect_dir, img_name.replace('.','_ori.'))
    merge_annoimg_path = os.path.join(detect_dir, img_name.replace('.','_merge4.'))

    cv2.imwrite(ori_annoimg_path, img)
    cv2.imwrite(merge_annoimg_path, img1)


    ## save  labels  as xml
    stitch_anno_name = os.path.basename(ori_path).split('.')[0] + save_ext
    save_path = os.path.join(stitch_dir, stitch_anno_name)
    if save_ext=='.xml':
        make_slice_voc(save_path, cluster_objs, sliceHeight=1024, sliceWidth=1024)
    else:
        make_slice_txt(save_path, cluster_objs)
    ## save_labels as txt




    # # NMS
    # pred = np.array(all_obj_list)
    # pred = torch.tensor(pred)
    # pred = torch.unsqueeze(pred,dim=0)
    # print(pred.shape)

    # pred = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.2)
    # pred = list(np.array(pred[0], dtype=np.float))
    # classes = ["component"]

    # new_pred = []
    # for i, p in enumerate(pred):
    #     p = list(p)
    #     cls = classes[int(p[-1])]
    #     new_pred.append([*p[:-1],cls]) 

    # print('***',new_pred[0])
    # print("all_crop_obj:",len(all_obj_list))
    # print('nms obj', len(new_pred))


    # img = ori_img.copy()
    # for obj in new_pred:
    #     box = obj[:4]
    #     img = box_label(img, box, label="comp")
    # cv2.imwrite('nms_stitch1.png', img)





def box_iou(box1, box2):
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=5000):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # # Cat apriori labels if autolabelling
        # if labels and len(labels[xi]):
        #     l = labels[xi]
        #     v = torch.zeros((len(l), nc + 5), device=x.device)
        #     v[:, :4] = l[:, 1:5]  # box
        #     v[:, 4] = 1.0  # conf
        #     v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
        #     x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # # Compute conf
        # x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        # box = xywh2xyxy(x[:, :4])
        box = x[:, :4]

        # # Detections matrix nx6 (xyxy, conf, cls)
        # if multi_label:
        #     i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
        #     x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        # else:  # best class only
        #     conf, j = x[:, 5:].max(1, keepdim=True)
        #     x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # # Filter by class
        # if classes is not None:
        #     x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS   ([n,4]) ([n])
        # i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output



if __name__ == "__main__":
    # crop_main()  
    # stitch_main()   

    stitch_anno_main()