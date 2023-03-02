from email.mime import base
import os
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
    for obj in root.findall('object'):
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

def exist_objs(slice_box,all_objs_list,sliceHeight, sliceWidth):
    '''
    slice_box:当前slice的图像边框
    all_objs_list:原图中的所有目标
    return:原图中于当前slice中的目标集合
    '''
    return_objs=[]
    min_h, min_w = 20, 20 #35, 35  # 有些目标GT会被窗口切分，太小的丢掉
    s_xmin, s_ymin, s_xmax, s_ymax = slice_box[0], slice_box[1], slice_box[2], slice_box[3]
    for vv in all_objs_list:
        category, xmin, ymin, xmax, ymax=vv[4],vv[0],vv[1],vv[2],vv[3]
        
        rec1 = vv[:4]
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

def bbox_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return: 
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积
 
    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s2# + s2 - a1
    iou = a1 / a2 #iou = a1/ (s1 + s2 - a1)
    return iou

def exist_objs_iou(list_1, list_2, sliceHeight, sliceWidth,win_h, win_w):
    # 根据iou判断框是否保留，并返回bbox
    return_objs=[]
    s_xmin, s_ymin, s_xmax, s_ymax = list_1[0], list_1[1], list_1[2], list_1[3]
    
    for single_box in list_2:
        xmin, ymin, xmax, ymax, category=single_box[0],single_box[1],single_box[2],single_box[3],single_box[4]
        iou = bbox_iou(list_1, single_box[:4])
        if iou > 0.2:
            if iou == 1:
                x_new=xmin-s_xmin
                y_new=ymin-s_ymin
                return_objs.append([x_new, y_new, x_new+(xmax-xmin), y_new+(ymax-ymin),category])
            else:
                xlist = np.sort([xmin, xmax, s_xmin, s_xmax])
                ylist = np.sort([ymin, ymax, s_ymin, s_ymax])
                #print(win_h, win_w, list_1, single_box, xlist[1] - s_xmin, ylist[1] - s_ymin)
                return_objs.append([xlist[1] - s_xmin, ylist[1] - s_ymin, xlist[2] - s_xmin, ylist[2] - s_ymin, category])
    return return_objs

def make_slice_voc(outpath,exiset_obj_list,sliceHeight=1024, sliceWidth=1024):
    name=outpath.split('/')[-1]
    #
    #
    with codecs.open(os.path.join(slice_voc_dir,  name[:-4] + '.xml'), 'w', 'utf-8') as xml:
        xml.write('<annotation>\n')
        xml.write('\t<filename>' + name + '</filename>\n')
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

def slice_im(base_images_dir, outdir, base_ann_dir, i=None, sliceHeight=640, sliceWidth=640,
             zero_frac_thresh=0.2, overlap=0.3, verbose=True):
    """
        从上到下从左到右进行切, 
        overlap:相邻两图的重叠度设置
    """
    cnt = 0
    emp_cnt = 0  # 记录切出来的没有元件的空图
    base_img_list = os.listdir(base_images_dir)

    # print(List_subsets)
    for per_img_name in tqdm(base_img_list):
        # print(per_img_name)
        # if 'c' not in per_img_name:
        #     continue
        out_name, _ = os.path.splitext(per_img_name)
        # out_name = str(out_name) + '_' + str(cnt)
        image_path = os.path.join(base_images_dir, per_img_name)
        ann_path = os.path.join(base_ann_dir, per_img_name[:-4] + '.xml')

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
        n_ims = 0
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
                #
                slice_xmax = x + sliceWidth
                slice_ymax = y + sliceHeight
                exiset_obj_list = exist_objs([x, y, slice_xmax, slice_ymax], 
                                                object_list, sliceHeight, sliceWidth)
                # exiset_obj_list = exist_objs_iou([x,y,slice_xmax,slice_ymax],object_list, sliceHeight, sliceWidth, win_h, win_w)
                
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
                        outpath = os.path.join(outdir,  out_name + '_crop' + str(cnt) + ext)
                        cnt += 1
                        # if verbose:
                        #     print("outpath:", outpath)
                        cv2.imwrite(outpath, window_c)
                        n_ims_nonull += 1
                        #------制作新的xml------
                        make_slice_voc(outpath, exiset_obj_list, sliceHeight, sliceWidth)

                else:
                    window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
                    emp_dir = '/media/zsl/data/zsl_datasets/for_train/717_base/base_crops/empty_imgs'
                    os.makedirs(emp_dir,exist_ok=True)
                    outpath = os.path.join(emp_dir,  out_name + '_empcrop' + str(emp_cnt) + ext)
                    #
                    emp_cnt += 1
                    cv2.imwrite(outpath, window_c)


def slice_im_random(base_images_dir, outdir, raw_ann_dir, i=None, sliceHeight=640, sliceWidth=640,
             zero_frac_thresh=0.2, times = 100, verbose=True):
    """
        随机切图
    """
    cnt = 0
    emp_cnt = 0
    base_img_list = os.listdir(base_images_dir)

    # print(List_subsets)
    for per_img_name in tqdm(base_img_list):
        # print(per_img_name)
        # if 'c' not in per_img_name:
        #     continue
        out_name, _ = os.path.splitext(per_img_name)
        # out_name = str(out_name) + '_' + str(cnt)
        image_path = os.path.join(base_images_dir, per_img_name)
        ann_path = os.path.join(raw_ann_dir, per_img_name[:-4] + '.xml')

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

        for i in range(times):
            x = int(random.randrange(0, win_w - sliceWidth + 1))
            y = int(random.randrange(0, win_h - sliceHeight + 1))
            slice_xmax = x + sliceWidth
            slice_ymax = y + sliceHeight
            exiset_obj_list = exist_objs([x, y, slice_xmax, slice_ymax], 
                                            object_list, sliceHeight, sliceWidth)
            # exiset_obj_list = exist_objs_iou([x,y,slice_xmax,slice_ymax],object_list, sliceHeight, sliceWidth, win_h, win_w)
            
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
                    outpath = os.path.join(outdir, out_name + \
                                           '|' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) + \
                                           '_' + str(pad) + ext)
                    outpath = os.path.join(outdir,  out_name + '_randomcrop' + str(cnt) + ext)
                    #
                    cnt += 1
                    # if verbose:
                    #     print("outpath:", outpath)
                    cv2.imwrite(outpath, window_c)
                    #------制作新的xml------
                    make_slice_voc(outpath, exiset_obj_list, sliceHeight, sliceWidth)
            else:
                window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
                emp_dir = './empty'
                os.makedirs(emp_dir,exist_ok=True)
                outpath = os.path.join(emp_dir,  out_name + '_Rempcrop' + str(emp_cnt) + ext)
                #
                emp_cnt += 1
                cv2.imwrite(outpath, window_c)




if __name__ == "__main__":
    """
        切训练集, 切图和切标签
    """
    # base_images_dir = '/media/zsl/data/zsl_datasets/for_train/scheme1/717_base/imgs_aug'   # 原始的图片
    # base_ann_dir = '/media/zsl/data/zsl_datasets/for_train/scheme1/717_base/labels_aug'
    # slice_voc_dir = '/media/zsl/data/zsl_datasets/for_train/scheme1/717_base/crop2/aug_labels'  # 
    # outdir = '/media/zsl/data/zsl_datasets/for_train/scheme1/717_base/crop2/aug_imgs'
    
    base_images_dir = '/media/zsl/data/zsl_datasets/for_train/717_base/base/imgs'  
    base_ann_dir = '/media/zsl/data/zsl_datasets/for_train/717_base/base/labels_xml'
    slice_voc_dir = '/media/zsl/data/zsl_datasets/for_train/717_base/base_crops_new/img_crop_labels'  # 
    outdir = '/media/zsl/data/zsl_datasets/for_train/717_base/base_crops_new/img_crops'

    if not os.path.exists(slice_voc_dir):
        os.makedirs(slice_voc_dir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    slice_im(base_images_dir, outdir,  base_ann_dir, sliceHeight=1024, sliceWidth=1024)
    # slice_im_random(base_images_dir, outdir, base_ann_dir, sliceHeight=1024, sliceWidth=1024)

