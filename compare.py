import os
import cv2
import time
import codecs
import xml.etree.ElementTree as ET
# from torch import R
from tqdm import tqdm
import shutil
from tqdm import trange                  # 显示进度条
from multiprocessing import cpu_count    # 查看cpu核心数
from multiprocessing import Pool

from patch_img import computer_IOU_conclude, is_conclude
import random
import numpy as np

from PIL import Image, ImageDraw, ImageFont
# from pathlib import Path


"""
    对比yolo检测得到的框结果
"""
classes = ["capacitor","resistor", "transistor",
            "ic", "pad", "inductor","others"]

def zooming(img, scale=0.1):
    img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    return img


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


def txt_parse(txt_path):
    r = []
    with open(txt_path, 'r') as f:  
        a = f.readlines()
        for l in a:
            l = l.strip()
            x = l.split(' ')
            ll =(x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]))
            r.append(ll)
        f.close()
    return r


def draw_one_label(img_path, label_path, save_dir,cls=None):
    img = cv2.imread(img_path)

    # roi_list = json_parse(label_path)
    roi_list = txt_parse(label_path)

    for roi in roi_list:
        # txt
        label = roi[0]
        x1,y1,x2,y2 = roi[1:]
        box = [x1,y1,x2,y2] 
        if cls is None:
            img = box_label(img, box, label=label)
        else:
            img = box_label(img, box, label=cls[int(label)])

    file_name = os.path.basename(img_path)
    file_path = os.path.join(save_dir,file_name)
    # print(file_path)
    cv2.imwrite(file_path, img)


# def _compare(t_comp_list, d_comp_list):
#     """
#         比较模板和测试板的检测框
#     """
#     match_list = []     # 框位置一致,标签一致
#     diff_label_list = []    # 框位置一致, 标签不一致
#     missPart_list = []      # 模板有框, 测试图没有框, 可能是少件
#     extraPart_list = []     # 模板没有框,测试图有框, 可能是多件
#     missAligned_list = []   # 模板框和测试框 重叠差异大, 可能是错件

#     matchest_comp_list = []
#     # get missPart
#     for t_comp in t_comp_list:
#         boxt = t_comp[1:]
#         labelt = t_comp[0]
#         # print(boxt, labelt)
#         max_iou = 0
#         matchest_comp = None
#         for d_comp in d_comp_list:
#             boxd = d_comp[1:]
#             labeld = d_comp[0]      
#             iou = compute_IOU(boxd, boxt)  
#             if iou > max_iou:
#                 max_iou = iou
#                 matchest_comp = d_comp
#         # 
#         if max_iou == 0:
#             missPart_list.append(t_comp)
#         #
#         elif max_iou > 0.5:
#             if labelt == matchest_comp[0]:
#                 match_list.append((t_comp, matchest_comp))
#             else:
#                 # print(labelt, labeld)
#                 diff_label_list.append((t_comp, matchest_comp))
#         else:
#             missAligned_list.append((t_comp, matchest_comp))

#         if matchest_comp is not None:
#             matchest_comp_list.append(matchest_comp)

#     # extraPart_list = [d_comp for d_comp in d_comp_list if d_comp not in matchest_comp_list]
        
#     for d_comp in d_comp_list:
#         boxd = d_comp[1:]
#         labeld = d_comp[0]
#         # print(boxt, labelt)
#         max_iou = 0
#         matchest_comp = None
#         for t_comp in t_comp_list:
#             boxt = t_comp[1:]
#             labeld = t_comp[0]      
#             iou = compute_IOU(boxd, boxt)  
#             if iou > max_iou:
#                 max_iou = iou
#                 matchest_comp = d_comp

#         if max_iou == 0:
#             extraPart_list.append(d_comp)


#     return  match_list, diff_label_list, missPart_list, extraPart_list, missAligned_list



def _compare(t_comp_list, d_comp_list):
    """
        比较模板和测试板的检测框
    """
    match_list = []     # 框位置一致,标签一致
    diff_label_list = []    # 框位置一致, 标签不一致
    missPart_list = []      # 模板有框, 测试图没有框, 可能是少件
    extraPart_list = []     # 模板没有框,测试图有框, 可能是多件
    missAligned_list = []   # 模板框和测试框 重叠差异大, 可能是错件

    assigned_list = []    # 测试图与模板匹配并分类的检测框.
    matchest_id = -1
    matchest_pair = [0]*len(d_comp_list)   # 匹配时测试板的每个框只能和一个模板框匹配

    assert len(t_comp_list) > 0
    assert len(d_comp_list) > 0

    for i, t_comp in enumerate(t_comp_list):
        boxt = t_comp[1:]
        labelt = t_comp[0]
        max_iou = 0
        matchest_comp = None
        for j, d_comp in enumerate(d_comp_list):

            boxd = d_comp[1:]
            labeld = d_comp[0]      
            iou, is_conclude = computer_IOU_conclude(boxd, boxt)  
            if iou >= max_iou and matchest_pair[j]==0:  #  且该框没有被最优匹配过
                matchest_id = j
                max_iou = iou
                matchest_comp = d_comp
        matchest_pair[matchest_id] = 1
        
        assigned_list.append(matchest_comp)  

        if max_iou == 0:
            missPart_list.append(t_comp)
        #
        elif max_iou > 0.5:
            if int(float(labelt)) == int(float(matchest_comp[0])):   # 临时 转
                match_list.append((t_comp, matchest_comp))
            else:
                diff_label_list.append((t_comp, matchest_comp))
        # iou 不大 但是是子集关系
        elif max_iou > 0.3 or is_conclude:
            if int(float(labelt)) == int(float(matchest_comp[0])):
                match_list.append((t_comp, matchest_comp))
            else:
                diff_label_list.append((t_comp, matchest_comp))
        else:
            missAligned_list.append((t_comp, matchest_comp))



    # 未被匹配并分配的为多件
    for d_comp in d_comp_list:
        if d_comp not in assigned_list:
            extraPart_list.append(d_comp)

    return  match_list, diff_label_list, missPart_list, extraPart_list, missAligned_list


def draw_boxes(img, comps, color=(255,0,0), label=False):
    """
        给框区域加蒙板
    """
    mask = np.zeros((img.shape), dtype=np.uint8)
    for comp in comps:
        cls = comp[0]
        comp_label = classes[int(float(cls))]
        box = list(comp[1:])
        box_points = get_box_points(box)
        mask = cv2.fillPoly(mask, [box_points], color=color)

        if label:
            label_h = 30
            x ,y = int(box[0]), int(box[1]+label_h)
            cv2.putText(img, comp_label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1,
                        color=color,
                        thickness=3,
                        lineType=0)
        
        center = (round(abs(box[2]+box[0])/2), round(abs(box[3]+box[1])/2))
        radius = int(max((box[2]-box[0]),(box[3]-box[1])))
        size = (radius,radius)
        # angle = np.random.randint(0, 361)
        colore_circle = (255,255,0)
        mask = cv2.ellipse(mask, center, size, 0, 0, 360, colore_circle, thickness=30)  #  (187)r (219)g (136)b

        # # 加框
        # bw = (box[2]-box[0])
        # bh = (box[3]-box[1])
        # pt1 = (int(box[0]-bw),int(box[1]-bh))
        # pt2 = (int(box[2]+bw),int(box[3]+bh))    
        # mask = cv2.rectangle(mask, pt1, pt2, color=(255,255,0),thickness=50,lineType=0)

    mask_img = 0.9 * mask + img

    return mask_img


def get_box_points(box):
    x_min, y_min, x_max, y_max = box
    points = [[x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min]
            ]
    points = np.array(points, dtype=np.int32)
    return points

def draw_compares(imgt, imgd, l, feature=None):
    """
        画出多件或者少件的对比图
    """
    mask_imgt = draw_boxes(imgt, l)
    mask_imgd = draw_boxes(imgd, l, (0,0,255))
    imgt_draw = zooming(mask_imgt, 0.5)
    imgd_draw = zooming(mask_imgd, 0.5)
    cat_img = np.concatenate((imgd_draw, imgt_draw), axis=1)

    return cat_img


def draw_match(imgt, imgd, l):
    """
        画共同有的对比图
    """
    t_comps = []
    d_comps = []
    for item in l:
        t_comp = item[0]
        d_comp = item[1]
        t_comps.append(t_comp)
        d_comps.append(d_comp)
 
    mask_imgt = draw_boxes(imgt, t_comps)
    mask_imgd = draw_boxes(imgd, d_comps)
    imgt_draw = zooming(mask_imgt, 0.5)
    imgd_draw = zooming(mask_imgd, 0.5)
    cat_img = np.concatenate((imgt_draw, imgd_draw), axis=1)

    return cat_img


def draw_title(img , title_text="Result", is_test=False, comps_lists=None):
    """
        对比图的标题
        title_text: 标题文字
        is_test : 标记该图是否为模板图
        comp_list : 
    """
    h, w, _ = img.shape
    title = np.ones((500, w, 3))*255

    # # FONT_HERSHEY_DUPLEX  FONT_ITALIC
    title = Image.fromarray(np.uint8(title))
    font1 = ImageFont.truetype("fonts/Font_platech.ttf",200)#设置字体类型和大小>
    draw = ImageDraw.Draw(title)
    draw.text((50,30), title_text, font=font1,fill=(0,0,0))

    if is_test:
        half_w = w/2
        font2 = ImageFont.truetype("fonts/Font_platech.ttf",70)#设置字体类型和大小

        draw.rectangle([half_w,100, half_w+100,200,],fill=(255,0,0))
        draw.text((half_w+200,100), "少件数量:", font=font2,fill=(0,0,0))

        draw.rectangle([half_w, 250, half_w+100,350,],fill=(0,0,255))
        draw.text((half_w+200,250), "多件数量:", font=font2,fill=(0,0,0))

        # 多件数量
        if comps_lists is not None:
            miss_num = len(comps_lists["missPart"])
            extra_num = len(comps_lists["extraPart"])

            draw.text((half_w+600,100), str(miss_num), font=font2,fill=(0,0,0))
            draw.text((half_w+600,250), str(extra_num), font=font2,fill=(0,0,0))


    title = np.array(title)
    title_img = np.concatenate((title, img),axis=0)

    return  title_img


def draw_all(imgt, imgd, comps_lists):
    """
        画出匹配件, 多件少件
        ls : {"missPart":[], "extraPart":[]}
    """
    feature = ["missPart", "extraPart", "missAlign", "diffLabel", "matches"]
    colors = [[(255,0,0),"Blue"], [(0,0,255),"Red"], [(0,255,0),"Green"], [(0,127,127),"Yellow"], [(255,255,255),"White"]]
    mask_imgt = imgt.copy()
    mask_imgd = imgd.copy()

    for f in comps_lists:
        if f in feature:
            color_index = feature.index(f)
            color, color_name=colors[color_index]
            print(f, color, color_name)

            if f == "matchest" or f == "diffLabel" or f == "missAlign":
                comp_l = comps_lists[f]
                comp_l_t = []
                comp_l_d = []
                for item in comp_l:
                    t_comp = item[0]
                    d_comp = item[1]
                    comp_l_t.append(t_comp)
                    comp_l_d.append(d_comp)
                mask_imgt = draw_boxes(mask_imgt, comp_l_t, color=color)
                mask_imgd = draw_boxes(mask_imgd, comp_l_d, color=color)  
            else:
                comp_l = comps_lists[f]
                mask_imgt = draw_boxes(mask_imgt, comp_l, color=color)
                mask_imgd = draw_boxes(mask_imgd, comp_l, color=color)

    imgt_draw = zooming(mask_imgt, 0.5)
    imgd_draw = zooming(mask_imgd, 0.5)

    imgt_draw = draw_title(imgt_draw, title_text="模板图", is_test=False)
    imgd_draw = draw_title(imgd_draw, title_text="测试图", is_test=True, comps_lists=comps_lists)

    cat_img = np.concatenate((imgd_draw, imgt_draw), axis=1)

    return cat_img


def compare_test():
    """
        比较模板和测试板的labels 
    """
    label_t_path = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_labeled/7.txt"
    label_d_path = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_anno_stitch_6wbig790000/template/7.txt"
    img_d_path = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_img_stitch_clean6w84_noconf/template/7_merge.jpg"
    img_t_path = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/template_imgs/7.jpg"
    save_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/ddddddddddd"
    os.makedirs(save_dir, exist_ok=True)

    t_comp_list = txt_parse(label_t_path)
    d_comp_list = txt_parse(label_d_path)
    img_d = cv2.imread(img_d_path)

    matchest_l, difflabel_l, misspart_l, extrapart_l, missalign_l = _compare(t_comp_list, d_comp_list)

    img_d = cv2.imread(img_d_path)
    img_t = cv2.imread(img_t_path)
    imgd_name, imgd_ext = os.path.splitext(os.path.basename(img_d_path))
    print("t box num:", len(t_comp_list))
    print("d box num:", len(d_comp_list))

    print("total match num:", len(matchest_l))
    print("match but label not num:", len(difflabel_l))
    print("missPart num:", len(misspart_l))
    print("missalign num:", len(missalign_l))
    print("extraPart num:", len(extrapart_l))

    # 一起框
    l_dic = {}
    l_dic["missPart"] = misspart_l
    l_dic["extraPart"] = extrapart_l
    # l_dic["matchest"] = matchest_l
    l_dic["missAlign"] = missalign_l
    l_dic["diffLabel"] = difflabel_l

    img_compare_draw = draw_all(img_t, img_d, l_dic)
    save_path = os.path.join(save_dir, imgd_name+"_all"+imgd_ext)
    cv2.imwrite(save_path, img_compare_draw)


def compare_one(label_t_path, label_d_path, img_t_path, img_d_path):
    """
        比较模板和测试板的labels ,返回比较图
    """
    t_comp_list = txt_parse(label_t_path)
    d_comp_list = txt_parse(label_d_path)

    img_d = cv2.imread(img_d_path)
    img_t = cv2.imread(img_t_path)

    matchest_l, difflabel_l, misspart_l, extrapart_l, missalign_l = _compare(t_comp_list, d_comp_list)
    print("t box num:", len(t_comp_list))
    print("d box num:", len(d_comp_list))
    print("total match num:", len(matchest_l))
    print("match but label not num:", len(difflabel_l))
    print("missPart num:", len(misspart_l))
    print("missalign num:", len(missalign_l))
    print("extraPart num:", len(extrapart_l))

    # 全部框
    l_dic = {}
    l_dic["missPart"] = misspart_l
    l_dic["extraPart"] = extrapart_l
    # l_dic["matchest"] = matchest_l
    l_dic["missAlign"] = missalign_l
    l_dic["diffLabel"] = difflabel_l

    img_compare_draw = draw_all(img_t, img_d, l_dic)

    return img_compare_draw
 

def batch_compare():
    """
        这里的label为 xyxy 非xywh
    """
    # label_t_dir = "/media/zsl/data/zsl_datasets/717/pro/test_lab/compare_test_scls/label_t"
    # label_d_dir = "/media/zsl/data/zsl_datasets/717/pro/test_lab/compare_test_scls/label_d"
    # img_d_dir = "/media/zsl/data/zsl_datasets/717/pro/test_lab/compare_test_scls/img_d_labeled"
    # img_t_dir = "/media/zsl/data/zsl_datasets/717/pro/test_lab/compare_test_scls/img_t_labeled"
    # save_dir = "/media/zsl/data/zsl_datasets/717/pro/test_lab/compare_test_scls/compare_results_scl2"

    label_t_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/1_compare2/template_anno_stitch_clean6w84/template"
    label_d_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/1_compare2/defects_aligned_whiteboard_stitch_anno_clean6w84"
    img_d_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/1_compare2/defects_aligned_whiteboard_stitch_img_clean6w84"
    img_t_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/1_compare2/template_img_stitch_clean6w84/template"
    save_dir = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/flow2/compare_results_scl_whiteboard_clean6w84_circle001"

    os.makedirs(save_dir, exist_ok=True)

    for cur_dir, sub_dir, files in os.walk(img_d_dir):
        if len(files) == 0:
            continue
        else:
            t_name = cur_dir.split('/')[-1]
            if t_name.find('3') < 0:
                continue
            save_sub_dir = os.path.join(save_dir, t_name)
            if not os.path.exists(save_sub_dir):
                os.makedirs(save_sub_dir, exist_ok=True)

            for file in tqdm(files):
                d_name, d_ext = os.path.splitext(file)
                if d_name.find('_ori') >= 0:   
                    continue
                else:
                    img_d_path = os.path.join(cur_dir, file)
                    print(t_name, d_name)

                    img_t_path = os.path.join(img_t_dir, t_name+'_merge.jpg')  #
                    label_t_path = os.path.join(label_t_dir, t_name+'.txt')

                    d_name = d_name.replace('_merge','')
                    label_d_path = os.path.join(label_d_dir, t_name, d_name+'.txt')
                    print(label_t_path)
                    print(label_d_path)
                    print(img_t_path)
                    print(img_d_path)
                    print("====")

                    img_compare_draw = compare_one(label_t_path, label_d_path, img_t_path, img_d_path)

                    imgd_name, imgd_ext = os.path.splitext(os.path.basename(img_d_path))
                    save_path = os.path.join(save_sub_dir, imgd_name+"_all"+imgd_ext)
                    cv2.imwrite(save_path, img_compare_draw)


if __name__ == "__main__":
    # compare_main()
    compare_test()
    # batch_compare()