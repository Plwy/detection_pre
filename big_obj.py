import os
import cv2
import glob
import numpy as np
import torch
import random

from compare import txt_parse
from stitch_test import box_label_yl5, make_slice_txt
from image_crop import compute_IR

"""
    以大目标为中心的随机切图.
    筛选大目标,得到pcba板mask,调整到指定尺寸按照阈值筛选大目标.
    以大目标为中心随机切图,切标签.
"""

def exist_objs(slice_box,all_objs_list):
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


def big_obj_center_crop(img, obj_list,  ori_big_box, img_name, crops_big_obj_img_dir, crops_big_obj_label_dir):
    """
        已知大目标框,在原图上以大目标为中心做随机切图.
        切图大小与目标框大小和在原图中的位置有关.
        同时切标签
    """
    h, w, c = img.shape
    bx_min, by_min, bx_max, by_max = ori_big_box

    im_name, ext = img_name.split('.')
    bx_suff = '|'+"_".join([str(int(a)) for a in ori_big_box])
    
    time = 3
    c = 0
    while c < time:
        c += 1

        # # # random crop size
        crop_x_min = random.randint(0, bx_min)   # 先随机生成左上角点
        crop_y_min = random.randint(0, by_min)

        bbw_min = bx_max - crop_x_min
        bbh_min = by_max - crop_y_min
        cropwh_min = max(bbw_min, bbh_min)      # box边长 最小要将big box 包含
        bsw = w-crop_x_min
        bsh = h-crop_y_min
        cropwh_max = min(bsw, bsh)  # 边长 最大不可超过图像边界
        print(cropwh_min, cropwh_max)

        if cropwh_min < cropwh_max:
            cropwh = random.randint(cropwh_min, cropwh_max)   # 再随机一个宽高值, 设置宽高相等, 否则生存的切图宽高比例差异过大
            crop_y_max = crop_y_min + cropwh
            crop_x_max = crop_x_min + cropwh
            crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max, :]
        else: #可能框在右下角,无法包括元件后切出一个正方形,先切出来再进行填充
            if bsh < cropwh_min:
                pad_bottom = int(cropwh_min - bsh)
                crop_y_max = h
            else:
                pad_bottom = 0
                crop_y_max = int(crop_y_min + cropwh_min)

            if bsw < cropwh_min:
                pad_right = int(cropwh_min - bsw)
                crop_x_max = w
            else:
                pad_right = 0
                crop_x_max = int(crop_x_min + cropwh_min)

            # crop img
            crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max, :]
            print(crop_img.shape, cropwh_min)
            border_color = (255, 255, 255)
            crop_img = cv2.copyMakeBorder(crop_img, 0, pad_bottom, 0, pad_right,
                                        cv2.BORDER_CONSTANT, value=border_color)
            print(crop_img.shape, cropwh_min)


        # crop labels
        crop_img_box = [crop_x_min, crop_y_min, crop_x_max, crop_y_max]
        crop_left_obj_list = exist_objs(crop_img_box, obj_list)   # class,... ==> ...,class


        bcrop_name = '_bigcenter'+str(c)
        # 构造保存的图像名称
        big_crop_img_path = os.path.join(crops_big_obj_img_dir, im_name+bx_suff+bcrop_name+'.'+ext)
        big_crop_label_path = os.path.join(crops_big_obj_label_dir, im_name+bx_suff+bcrop_name+'.txt')

        print('======big_crop_img_path:',big_crop_img_path)
        print('======big_crop_label_path',big_crop_label_path)

        cv2.imwrite(big_crop_img_path, crop_img)
        make_slice_txt(big_crop_label_path ,crop_left_obj_list)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
        from yolov5 aug resize and pad
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def letterbox_test():
    img_path = "/home/zsl/Pictures/Wallpapers/francesco-ungaro-1fzbUyzsHV8-unsplash.jpg"
    img = cv2.imread(img_path)
    print(img.shape)
    x, r, d  = letterbox(img)
    cv2.imshow('x',x)
    cv2.waitKey(0)
    print(x.shape, r)
    print(d)



def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
      
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def draw_min_rect_rectangle(mask, image):
    """
        切出mask的白色最大外接矩形
    """
    # thresh = cv2.Canny(mask, 128, 256)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = np.copy(image)
    
    print("contours len", len(contours))
    # for i, cnt in enumerate(contours):
    #     # x, y, w, h = cv2.boundingRect(cnt)
    #     # # 绘制矩形
    #     # # cv2.rectangle(img,  (x, y+h), (x+w, y), (0, 255, 255))
    #     # crop_img = img[y:y+h, x:x+w, :]
    #     # crop_box = [x,y,w,h]
    max = 0
    max_area = 0
    for i, c in enumerate(contours):
        c_a = cv2.contourArea(c)
        # c_a = c.shape[0]
        if c_a > max_area:
            max = i
            max_area = c_a

    max_c = contours[max]
    x, y, w, h = cv2.boundingRect(max_c)
    crop_img = img[y:y+h, x:x+w, :]
    crop_box = [x,y,w,h]

    return crop_img, crop_box


def get_mask(img):
    """
        获取pcba板的mask.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = img_hsv[:,:,1] 
    _, mask = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((40,40),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max = 0
    max_area = 0
    for i, c in enumerate(contours):
        # c_a = cv2.contourArea(c)
        c_a = c.shape[0]
        if c_a > max_area:
            max = i
            max_area = c_a

    max_c = contours[max]
    points = max_c[:,0,:]
    maskx = np.zeros_like(img_gray)
    maskx = cv2.fillPoly(maskx, [points], color=(255,255,255))

    return maskx


def get_t_mask(imgt):
    """
        获取模板图pcba板区域的mask
    """
    # imgt_p = "test/zsl_test_image/template/1.jpg"
    # imgt = cv2.imread(imgt_p)
    imgt_hsl = cv2.cvtColor(imgt, cv2.COLOR_BGR2HLS)
    t_l = imgt_hsl[:,:,2]
    _, t_l = cv2.threshold(t_l,30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((37,37),np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55))  
    t_l = cv2.morphologyEx(t_l, cv2.MORPH_OPEN, kernel)
    # temp = t_l.copy()
    kernel = np.ones((37,37),np.uint8)

    t_l = cv2.morphologyEx(t_l, cv2.MORPH_CLOSE, kernel)
    contours , h = cv2.findContours(t_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max = 0
    max_c = []
    for c in contours:
        c_a = cv2.contourArea(c)
        if c_a > max:
            max = c_a
            max_c.append(c)
    print(max_c[-1].shape)
    b_c = max_c[-1]

    # b_c_ps = cv2.convexHull(b_c.astype(int))
    # cv2.fillPoly(mask, b_c_ps, (255,255,255),-1)
    mask = np.zeros_like(imgt)
    t_h,t_w,_ = imgt.shape
    for w in range(t_w):
        for h in range(t_h):
            result = cv2.pointPolygonTest(b_c, (w,h), False)
            if result >= 0:
                mask[h,w,:] = (255,255,255)
            else:
                mask[h,w,:] = (0,0,0)
    # mask = zooming(mask, 0.2)

    return mask



def batch_crop_mask():
    """
        依照mask最大外接矩形, 切出pcba板, 同时重写label
        resize到 640 挑出边框大于40的元件
    """
    mask_dir = "/media/zsl/data/zsl_datasets/PCB_test/big_obj/template_mask_good"
    img_dir  = "/media/zsl/data/zsl_datasets/PCB_test/big_obj/template_imgs"
    label_ori_dir = "/media/zsl/data/zsl_datasets/PCB_test/big_obj/template_labels_txt_ori"

    crop_label_dir = "/media/zsl/data/zsl_datasets/PCB_test/big_obj/template_crop_labels"
    crop_img_dir = "/media/zsl/data/zsl_datasets/PCB_test/big_obj/template_crop_imgs"
    resize_img_dir = "/media/zsl/data/zsl_datasets/PCB_test/big_obj/template_resize_imgs"


    os.makedirs(crop_label_dir, exist_ok=True)
    os.makedirs(crop_img_dir, exist_ok=True)
    os.makedirs(resize_img_dir, exist_ok=True)

    path_list = glob.glob(mask_dir+'/*')
    print(len(path_list))
    
    for mask_path in path_list:
        mask = cv2.imread(mask_path)

        img_name = os.path.basename(mask_path)
        img_path = os.path.join(img_dir, img_name)

        image = cv2.imread(img_path)

        img_crop, crop_box = draw_min_rect_rectangle(mask, image)

        #获取原始的txt label
        label_ori_path = os.path.join(label_ori_dir, img_name.replace('.jpg', '.txt'))
        print("ori label path",label_ori_path)
        obj_list = txt_parse(label_ori_path)  # [0, xyxy]

        # check ori box
        img = image.copy()
        for obj in obj_list:
            box = obj[1:]
            cls = str(obj[0])
            img = box_label_yl5(img, box)
        # cv2.imwrite('t.png', img)

        # to write new box
        crop_label_path = os.path.join(crop_label_dir,  img_name.replace('.jpg', '.txt'))
        label_txt = open(crop_label_path, 'w')
        slice_x, slice_y, slice_w, slice_h = crop_box

        #save crop labels
        crop = img_crop.copy()  # for check new box

        croped_objs = []     #
        crop_bbox = []
        for obj in obj_list:
            x_min = float(obj[1])-slice_x
            x_max = float(obj[3])-slice_x
            y_min = float(obj[2])-slice_y
            y_max = float(obj[4])-slice_y
            bbox = [x_min, y_min, x_max, y_max]
            classid = str(obj[0])
            crop_bbox.append(bbox)
            croped_objs.append([obj[0], x_min, y_min, x_max, y_max])
            label_txt.write(classid + " " + " ".join([str(a) for a in bbox]) + '\n')
            crop = box_label_yl5(crop, bbox)   # check
        # cv2.imwrite('crop.png', crop)
        label_txt.close()

        # save crop img
        # crop_img_path = os.path.join(crop_img_dir,  img_name)
        # # cv2.imwrite(crop_img_path, img_crop)
        resize_img_path = os.path.join(resize_img_dir, img_name)
        cv2.imwrite(resize_img_path, crop)



        #################
        ##   
        print("1",img_crop.shape)

        img_resize,r,ds = letterbox(img_crop)

        print("2",img_resize.shape)
        print("3",r, ds)
        # xh = (img_resize.shape[0]-ds[1])/r[0]
        # xw = (img_resize.shape[1]-ds[0])/r[1]
        # print("4", xh,xw)


        det = np.array(crop_bbox)
        print(det[:3])

        det[:, :4] = scale_coords(img_crop.shape, det[:, :4], img_resize.shape).round()

        det = list(det)
        print(len(det), det[:3])

        img_rr = img_resize.copy()
        for box in det:
            x_min,y_min,x_max,y_max=box
            bw = x_max-x_min
            bh = y_max-y_min
            if bw > 40 or bh > 40:
                img_rr = box_label_yl5(img_rr, box)   # check
        resize_img_path = os.path.join(resize_img_dir, img_name)
        cv2.imwrite(resize_img_path, img_rr)


def main():
    """
    - 阈值划分得到pcba板的mask
    - 得到板mask的最小外接矩形.切出来. (标注同时切)
    - resize到640*640 , 边界填充,设置阈值获得大目标.
    """
    big_box_thred = 25
    img_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/base/imgs"   # 待切图路径
    label_ori_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/base/labels_txt"   # 待切图标签

    # 设置最终切好的大目标存储路径
    crops_big_obj_img_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/1_big_obj_crops_imgs" # 存储随机切出的大目标图
    crops_big_obj_label_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/1_big_obj_crops_labels" # 存储大件图对应的标签
    os.makedirs(crops_big_obj_img_dir, exist_ok=True)
    os.makedirs(crops_big_obj_label_dir, exist_ok=True)

    # 设置四个中间值的存储路径
    mask_dir = '/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/1_mask'  # 存得到的pcba班的mask结果
    crop_label_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/1_maskcrop_labels" # 存根据mask切图后的标签
    crop_img_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/1_maskcrop_imgs" # 存根据mask切图后的图
    resize_img_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/1_maskcropresize_imgs" # 存根据mask切图后再resize的图
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(crop_label_dir, exist_ok=True)
    os.makedirs(crop_img_dir, exist_ok=True)
    os.makedirs(resize_img_dir, exist_ok=True)


    path_list = glob.glob(img_dir+'/*')
    print(len(path_list))
    for img_path in path_list:
        print(img_path)

        image = cv2.imread(img_path)
        
        # 1.get mask
        mask = get_mask(image)   

        # save mask
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, img_name)
        cv2.imwrite(mask_path, mask)

        # 2.crop mask
        img_crop, crop_box = draw_min_rect_rectangle(mask, image)
        crop_img_path = os.path.join(crop_img_dir, img_name)
        cv2.imwrite(crop_img_path, img_crop)

        # get ori txt label
        label_ori_path = os.path.join(label_ori_dir, img_name.replace('.jpg', '.txt'))
        obj_list = txt_parse(label_ori_path)  # [0, xyxy]

        # # check ori box
        # img = image.copy()
        # for obj in obj_list:
        #     box = obj[1:]
        #     cls = str(obj[0])
        #     img = box_label_yl5(img, box)
        # # cv2.imwrite('t.png', img)


        # 3. get crop box label
        crop_label_path = os.path.join(crop_label_dir,  img_name.replace('.jpg', '.txt'))
        label_txt = open(crop_label_path, 'w')
        slice_x, slice_y, slice_w, slice_h = crop_box

        # save crop labels
        # crop = img_crop.copy()  # for check new box
        crop_bbox = []     #
        crop_bbox_index = []   # record box ori obj_list index
        for i, obj in enumerate(obj_list):
            x_min = float(obj[1])-slice_x
            x_max = float(obj[3])-slice_x
            y_min = float(obj[2])-slice_y
            y_max = float(obj[4])-slice_y
            bbox = [x_min, y_min, x_max, y_max]
            classid = str(obj[0])
            crop_bbox_index.append([bbox,i])
            crop_bbox.append(bbox)
            label_txt.write(classid + " " + " ".join([str(a) for a in bbox]) + '\n')
            # crop = box_label_yl5(crop, bbox)   # check
        # cv2.imwrite('crop.png', crop)
        label_txt.close()

        # 4. resize and pad
        img_resize,r,ds = letterbox(img_crop)
        det = np.array(crop_bbox)

        # det[:, :4] = scale_coords(img_crop.shape, det[:, :4], img_resize.shape).round()
        det[:, :4] = scale_coords(img_crop.shape, det[:, :4], img_resize.shape)

        det = list(det)


        img_rr = img_resize.copy()
        img_ori = image.copy()
        big_boxes = []
        ori_obj = []
        bigw = 0
        bigh = 0
        big_cnt = 0 ## big obj count
        for i, box in enumerate(det):
            x_min,y_min,x_max,y_max=box
            bw = x_max-x_min
            bh = y_max-y_min
            if bw > big_box_thred or bh > big_box_thred:
                big_cnt += 1
                # img_rr = box_label_yl5(img_rr, box)   # check
                big_boxes.append(box)
                ori_index = crop_bbox_index[i][1]
                ori_big_box = obj_list[ori_index][1:]     # big box in ori image
                # img_ori = box_label_yl5(img_ori, ori_big_box)   # check

                # # crop and save
                big_obj_center_crop(img_ori, obj_list,  ori_big_box, 
                                    img_name, crops_big_obj_img_dir, 
                                    crops_big_obj_label_dir)

                ori_bw = ori_big_box[2]-ori_big_box[0]
                ori_bh = ori_big_box[3]-ori_big_box[1]
                if ori_bw > bigw:
                    bigw = ori_bw
                if ori_bh > bigh:
                    bigh = ori_bh

        print("max obj w:", bigw)
        print("max obj h:", bigh)
        print("big count:", big_cnt)
        resize_img_path = os.path.join(resize_img_dir, img_name)
        cv2.imwrite(resize_img_path, img_resize)



if __name__ == "__main__":
    # letterbox_test()
    main()
