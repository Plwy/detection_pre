from pickletools import uint8
import shutil
from cv2 import boxPoints
from matplotlib.pyplot import axis
import torch
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import codecs
from math import fabs, sin, radians, cos
import random

from stitch_test import make_slice_txt, txt_parse_conf

    
################

def txt_parse(txt_path):
    r = []
    with open(txt_path, 'r') as f:  
        a = f.readlines()
        for l in a:
            l = l.strip()
            x = l.split(' ')
            ll =(float(x[1]), float(x[2]), float(x[3]), float(x[4]), x[0])
            r.append(ll)
        f.close()
    return r



def xml_parse(xml_path):
    tree = ET.parse(xml_path)
    labels = {}
    for obj in tree.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)
        print(x_min,x_max,y_min,y_max)


def make_slice_voc(outpath, exiset_obj_list, sliceHeight=1024, sliceWidth=1024):
    """
        exiset_obj_list内的标签写入outpath
    """
    name=outpath.split('/')[-1]
    with codecs.open(os.path.join(outpath), 'w', 'utf-8') as xml:
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


def _elastic(image, p, alpha=None, sigma=None, random_state=None):
    """
    弹性变化是对像素点各个维度产生(-1，1)区间的随机标准偏差，并用高斯滤波（0，sigma）
    对各维度的偏差矩阵进行滤波，最后用放大系数alpha控制偏差范围。因而由A(x,y)得到
    的A’(x+delta_x,y+delta_y)。A‘的值通过在原图像差值得到，A’的值充当原来A位置
    上的值。一般来说，alpha越小，sigma越大，产生的偏差越小，和原图越接近。
    """
    if random.random() > p:
        return image
    if alpha is None:
        alpha = image.shape[0] * random.uniform(0.5, 2)
    if sigma is None:
        sigma = int(image.shape[0] * random.uniform(0.5, 1))
    # 随机种子
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[:2]
    dx, dy = [cv2.GaussianBlur((random_state.rand(*shape)*2-1) * alpha,
                               (sigma | 1, sigma | 1), 0) for _ in range(2)]

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x, y = np.clip(x+dx, 0, shape[1]-1).astype(np.float32), np.clip(y+dy, 0, shape[0]-1).astype(np.float32)
    return cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR, borderValue=0, borderMode=cv2.BORDER_REFLECT)


def random_flip_horizon(image, boxes, thre=0):
    """
        水平翻转
    """
    _, w, _ = image.shape

    if np.random.random() > thre:
        # print("flip hor!!!!!!!!!!!!!!")
        image = image[:, ::-1]   
        boxes = boxes.copy()
        boxes = np.array(boxes)
        boxes[:, 0::2] = w - boxes[:, 2::-2]
        boxes = list(boxes)
    return image, boxes


def random_flip_vertical(image, boxes, thre=0):
    """
        垂直翻转,改boxes
        input boxes 为list [[x_min,y_min,x_max,y_max],[...],[...]]
    """
    h, _, _ = image.shape
    if np.random.random() > thre:
        # print("flip Ver!!!!!!!!!!!!!!")
        image = image[::-1, :]
        boxes = boxes.copy()
        boxes = np.array(boxes)
        boxes[:, 1::2] = h - boxes[:, 3::-2]
        boxes = list(boxes)
    return image, boxes


def rotate_with_points(image, points, degree, fill=0):
    """逆时针旋转图像image角度degree并计算原图中坐标点points在旋转后的图像中的位置坐标.
    Args:
        image: 图像数组
        degree: 旋转角度
        points (np.array): ([x, ...], [y, ...]), shape=(2, m)，原图上的坐标点
        fill: 旋转图像后空白处填充的颜色默认填0
    Return:
        new_img: 旋转后的图像
        new_pts: 原图中坐标点points在旋转后的图像中的位置坐标
    """
    h, w,_ = image.shape

    new_h = int(w * fabs(sin(radians(degree))) + h * fabs(cos(radians(degree))))
    new_w = int(h * fabs(sin(radians(degree))) + w * fabs(cos(radians(degree))))

    rtt_mat = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)

    rtt_mat[0, 2] += (new_w - w) / 2
    rtt_mat[1, 2] += (new_h - h) / 2

    new_img = cv2.warpAffine(image, rtt_mat, (new_w, new_h), borderValue=fill)

    a = np.array([rtt_mat[0][: 2], rtt_mat[1][: 2]])
    b = np.array([[rtt_mat[0][2]], [rtt_mat[1][2]]])

    new_pts = np.round(np.dot(a, points.astype(np.float32)) + b).astype(np.int64)
    return new_img, new_pts

def rotate_image(image, degree, fill=0):
    """逆时针旋转图像image角度degree
    Return:
        new_img: 旋转后的图像
    """
    h, w,_ = image.shape

    new_h = int(w * fabs(sin(radians(degree))) + h * fabs(cos(radians(degree))))
    new_w = int(h * fabs(sin(radians(degree))) + w * fabs(cos(radians(degree))))

    rtt_mat = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)

    rtt_mat[0, 2] += (new_w - w) / 2
    rtt_mat[1, 2] += (new_h - h) / 2

    # new_img = cv2.warpAffine(image, rtt_mat, (new_w, new_h), borderValue=fill)
    new_img = cv2.warpAffine(image, rtt_mat, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))


    a = np.array([rtt_mat[0][: 2], rtt_mat[1][: 2]])
    b = np.array([[rtt_mat[0][2]], [rtt_mat[1][2]]])

    return new_img, a, b


def Rotate(image, boxes, degree=90, thre=0):
    """
        input:
            degree 为旋转角度, eg:90
    """
    if np.random.random() > thre:
        boxes = np.array(boxes)
        boxes = boxes.T

        points_lup = boxes[:2] # 左上角点
        points_rdown = boxes[2:] # 右下角点
        points_ldown = boxes[0::3]  # 左下角点
        points_rup = boxes[2::-1][:2] # 右上角点

        img_rot, a, b = rotate_image(image, degree)
        # 旋转左上角点,和右下角点
        new_pts_lup = np.round(np.dot(a, points_lup.astype(np.float32)) + b).astype(np.int64)
        new_pts_rdown = np.round(np.dot(a, points_rdown.astype(np.float32)) + b).astype(np.int64)
        new_pts_ldown = np.round(np.dot(a, points_ldown.astype(np.float32)) + b).astype(np.int64)
        new_pts_rup = np.round(np.dot(a, points_rup.astype(np.float32)) + b).astype(np.int64)

        # 重构box
        new_boxes = []
        for i in range(new_pts_lup.shape[1]):
            x1, y1 = new_pts_lup[0][i], new_pts_lup[1][i]
            x2, y2 = new_pts_rdown[0][i], new_pts_rdown[1][i]
            x3, y3 = new_pts_ldown[0][i], new_pts_ldown[1][i]
            x4, y4 = new_pts_rup[0][i], new_pts_rup[1][i]
            x_min = min(x1, x2, x3, x4)
            x_max = max(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            y_max = max(y1, y2, y3, y4)
            new_box = [x_min, y_min, x_max, y_max]
            new_boxes.append(new_box)
    else:
        img_rot, new_boxes = image, boxes
        
    return img_rot, new_boxes

def random_HSV(img, thre=0):
    """
        转HSV
    """
    if np.random.random()> thre:
        img_aug = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        img_aug = img
    return img_aug


def random_bright(img, u=60, thre=0):
    """
        改变亮度
    """
    if np.random.random() > thre:
        img = img.astype(np.float)
        alpha=np.random.uniform(-u, u)/255
        img+=alpha
        img = np.clip(img, a_min=0, a_max=255)
    return img

def random_contrast(img, lower=0.5, upper=1.5, thre=0):
    """
        改变对比度
    """
    if np.random.random() > thre:
        img = img.astype(np.float)
        alpha=np.random.uniform(lower, upper)
        img*=alpha
        img = np.clip(img, a_min=0, a_max=255)
    return img

def random_saturation(img, lower=0.5, upper=1.5, thre=0):
    """
        改变饱和度
    """
    if np.random.random() > thre:
        img = img.astype(np.float)
        alpha=np.random.uniform(lower, upper)
        img[1]=img[1]*alpha
        img = np.clip(img, a_min=0, a_max=255)
    return img
        
def add_gasuss_noise(img, mean=0, std=0.1):
    """
        添加高斯噪声  bug
    """
    # noise=torch.normal(mean,std,img.shape)
    img = img.astype(np.float)
    noise=np.random.normal(mean, std, img.shape)
    img+=noise
    img = np.clip(img, a_min=0, a_max=255)
    return

def add_salt_noise(img):
    # noise=torch.rand(img.shape)
    img = img.astype(np.float)
    noise = np.random.rand((img.shape))
    alpha=np.random.random()
    img[noise[:,:,:]>alpha]=1.0
    return img

def add_pepper_noise(img):
    # noise=torch.rand(img.shape)
    img = img.astype(np.float)
    noise = np.random.randint(img.shape)
    print("noise:", noise.shape)
    alpha=np.random.random()
    img[noise[:,:,:]>alpha]=0
    return img

def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()
    # 随机变换亮度
    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    # 随机变换对比度
    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))
    # 将图片转到HSV空间
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 随机色度变换
    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    # 随机变换饱和度
    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image




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

def zooming(img, scale=0.1):
    img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    return img

def test():
    img_path = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/444/1|1718_3780_2891_4196_bigcenter2.jpg"
    label_path = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/4442/1|1718_3780_2891_4196_bigcenter2.txt"

    image = cv2.imread(img_path)
    # tree = ET.parse(label_path)
    # labels = []
    # for obj in tree.findall('object'):
    #     name = obj.find('name').text
    #     bbox = obj.find('bndbox')
    #     x_min = int(bbox.find('xmin').text)
    #     y_min = int(bbox.find('ymin').text)
    #     x_max = int(bbox.find('xmax').text)
    #     y_max = int(bbox.find('ymax').text)
    #     labels.append([x_min,y_min,x_max,y_max,name])

    # boxes = []
    # for k in labels:
    #     boxes.append(k[:4])

    # # image_aug, boxes = random_flip_vertical(image, boxes)
    # print("#:", len(boxes), boxes[0])
    # image_aug, boxes_aug  = Rotate(image, boxes, degree=30)
    # print("##",len(boxes_aug), boxes_aug[0])
    # # #  对增强的测试
    # image = zooming(image,0.1)
    # image = random_bright(image)
    # cv2.imwrite('test_bright.png', image)
    # image = random_contrast(image)
    # cv2.imwrite('test_contrast.png', image)
    # image = random_saturation(image)
    # cv2.imwrite('test_saturation.png', image)
    image = add_gasuss_noise(image)
    cv2.imwrite('test_gasuss_noise.png', image)

    
    # image_aug = image_aug.astype(np.uint8)
    # for box in boxes_aug:
    #     image_aug = box_label(image_aug, box)
    # cv2.imwrite('draw_test_rot.png', image_aug)


def batch_aug_main():
    """
        遍历每个图, 随机选择进行增强.
        每张图都进行上下/左右/上下左右增强
        每张图都进行亮度或者对比度或者饱和度的增强
        进行一种噪声增强
        生成的图像数 = 基类图数目*选择的增强类别数
    """
    ori_img_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/base_crops_new/img_crops"
    ori_label_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/base_crops_new/img_crop_labels_txt"
    dst_img_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/val/img_val"
    dst_label_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/val/label_val"

    # ori_img_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/base_crops_new/img_crops"
    # ori_label_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/base_crops_new/img_crop_labels_txt"
    # dst_img_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/base_crops_new/img_crops_aug"
    # dst_label_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/base_crops_new/img_crop_labels_aug"


    # ori_img_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop_new/1_big_obj_crops_imgs"
    # ori_label_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop_new/1_big_obj_crops_labels"
    # dst_img_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop_new/1_big_obj_crops_imgs_augs"
    # dst_label_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop_new/1_big_obj_crops_labels_augs"


    label_ext = '.txt'   # 标签的类型

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    img_paths = glob.glob(ori_img_dir+'/*')
    aug_num = 1 # 每张图做几次增强
    n = 0
    for img_path in tqdm(img_paths):
        if n > 2000:
            break
        n += 1
        img = cv2.imread(img_path)
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        
        # xml
        if label_ext.find('.xml') >= 0:
            label_path = os.path.join(ori_label_dir, img_name+'.xml')
            print(label_path)
            tree = ET.parse(label_path)
            labels = []
            for obj in tree.findall('object'):
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                x_min = int(bbox.find('xmin').text)
                y_min = int(bbox.find('ymin').text)
                x_max = int(bbox.find('xmax').text)
                y_max = int(bbox.find('ymax').text)
                labels.append([x_min,y_min,x_max,y_max,name])


        else:   #.txt
            label_path = os.path.join(ori_label_dir, img_name+'.txt')
            print(label_path)
            labels = txt_parse(label_path)   # [..., class]


        boxes = []
        for k in labels:
            boxes.append(k[:4])

        aug_list = ["FlipHorizon", "FlipVertical","Rotate90",
                    "Rotate180", "Rotate270", "Rotate30","RotateRandom",
                    "HSV", "Bright","Contrast", "Saturation", "GaussNoise",
                    "SaltNoise","PepperNoise"]

        aug_k = [0,1,2,3,4,7,8,9,10]
        # aug_k = [6,11,12,13]
        aug_k = random.sample(aug_k, aug_num)
        print([aug_list[i] for i in aug_k])
        for k in aug_k:
            aug_ext = aug_list[k]
            boxes_aug = boxes
            if aug_ext == "FlipHorizon":
                img_aug, boxes_aug = random_flip_horizon(img, boxes_aug, thre=0)
            elif aug_ext == "FlipVertical":
                img_aug, boxes_aug = random_flip_vertical(img, boxes_aug, thre=0)
            elif aug_ext == "Rotate90":
                img_aug, boxes_aug = Rotate(img, boxes_aug, degree=90)
            elif aug_ext == "Rotate180":
                img_aug, boxes_aug = Rotate(img, boxes_aug, degree=180)
            elif aug_ext == "Rotate270":
                img_aug, boxes_aug = Rotate(img, boxes_aug, degree=270)
            elif aug_ext == "Rotate30":
                img_aug, boxes_aug = Rotate(img, boxes_aug, degree=30)
            elif aug_ext == "Rotate30":
                degree = random.randint(0,360)
                img_aug, boxes_aug = Rotate(img, boxes_aug, degree=degree)
            elif aug_ext == "HSV":
                img_aug = random_HSV(img, thre=0)
            elif aug_ext == "Bright":
                img_aug = random_bright(img)
            elif aug_ext == "Contrast":
                img_aug = random_contrast(img)
            elif aug_ext == "Saturation":
                img_aug = random_saturation(img)
            elif aug_ext == "GaussNoise":
                img_aug = add_gasuss_noise(img)
            elif aug_ext == "SaltNoise": 
                img_aug = add_salt_noise(img)
            elif aug_ext == "PepperNoise":
                img_aug = add_pepper_noise(img)
            else:
                img_aug = img

            # img_aug save
            img_aug_name = img_name+'_'+aug_ext + ext
            img_aug_path = os.path.join(dst_img_dir, img_aug_name)
            print("img_aug_path:",img_aug_path)
            cv2.imwrite(img_aug_path, img_aug)

            #new label   [xyxy,c] [xyxy,c,conf]
            aug_label = []
            for i in range(len(labels)):
                c = labels[i][-1] # boxs boxs[0] 
                box = list(boxes_aug[i])
                box.append(c)
                aug_label.append(box)
            
            # print("===",aug_label[:3])
            if label_ext.find('.xml') >= 0:

                label_aug_name = img_name+'_'+aug_ext+'.xml'
                aug_label_path = os.path.join(dst_label_dir, label_aug_name)
                print(aug_label_path)
                h,w,c = img_aug.shape
                make_slice_voc(aug_label_path, aug_label, 
                                sliceHeight=h,
                                sliceWidth=w)
            else:  # .txt
                label_aug_name = img_name+'_'+aug_ext+'.txt'
                aug_label_path = os.path.join(dst_label_dir, label_aug_name)
                print(aug_label_path)   
                make_slice_txt(aug_label_path, aug_label)


if __name__ == "__main__":
    batch_aug_main()
    # test()
