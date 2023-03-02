import os
import cv2
import json
from tqdm import tqdm
import codecs
import glob

import xml.etree.ElementTree as ET

from parse_objfile import json_parse
from parse_objfile import txt_parse


"""
json2xml()
json2yolo()
txt2xml()

"""

def xyxy2xywh(size, box):
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


def obj_xml_write(outpath, obj_list, sliceHeight=1024, sliceWidth=1024):
    """
        obj_list内的标签写入outpath
        obj_list:[[cls,x_min,y_min,x_max,y_max],...]  cls 为 str

    """
    name=outpath.split('/')[-1]
    print(name)
    with codecs.open(outpath, 'w', 'utf-8') as xml:
        xml.write('<annotation>\n')
        xml.write('\t<filename>' + name + '</filename>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(sliceWidth) + '</width>\n')
        xml.write('\t\t<height>' + str(sliceHeight) + '</height>\n')
        xml.write('\t\t<depth>' + str(3) + '</depth>\n')
        xml.write('\t</size>\n')
        cnt = 1
        for obj in obj_list:
            #
            bbox = obj[1:]
            class_name = obj[0]
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
            ### obj: [cls,x_min,y_min,x_max,y_max]
            # bbox = obj[1:]    
            # class_name = obj[0]  
            ### obj: [x_min,y_min,x_max,y_max,cls]
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


def json2yolo(json_path, txt_path, label_all_dic):
    """
       label_all_dic
        
    """
    roi_list = []
    with open(json_path, 'r') as f:
        #开始写txt
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
            x1,y1= point[0]
            x2,y2 = point[1]
            x_min = min(x1,x2)
            x_max = max(x1,x2)
            y_min = min(y1,y2)
            y_max = max(y1,y2)
            b = (float(x_min), float(x_max), float(y_min), float(y_max))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = xyxy2xywh((w, h), b)
            label_txt.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

        label_txt.close()



def json2xml_main():
    ori_json_dir = "/media/zsl/data/zsl_datasets/for_train/pcba_dd/label_base_nonoise"
    dst_xml_dir = "/media/zsl/data/zsl_datasets/for_train/pcba_dd/base_resize_xml"
    os.makedirs(dst_xml_dir, exist_ok=True)
    json_list = os.listdir(ori_json_dir)

    for json_file in json_list:
        json_path = os.path.join(ori_json_dir,json_file)
        print("==json_path==",json_path)
        labels,h,w = json_parse(json_path)

        filename, _ = json_file.split('.')
        xml_path = os.path.join(dst_xml_dir, filename+'.xml')
        print(xml_path)
        make_slice_voc(xml_path, labels, 
                        sliceHeight = h,
                        sliceWidth = w)


def txt2xml_main():
    """
        读取当前目录的文件,转化为其他格式,或进行更改
        txt format: [cls_id, x_min,y_min,x_max,y_max]
    """
    # 原始文档目录
    ori_dir = "/media/zsl/data/zsl_datasets/717/pro/+test_lab/result2w_all_t/exp/labels"
    # 更新文档目录
    output_dir = "/media/zsl/data/zsl_datasets/717/pro/test_lab/result2w_all_t/exp/labels_xml"
    os.makedirs(output_dir, exist_ok=True)
    ori_ext = '.txt'
    dst_ext = '.xml'
    ori_paths = glob.glob(ori_dir + '/*' + ori_ext)
    classes = ["component"]
    for ori_path in tqdm(ori_paths): 
        # dst_file = os.path.basename(ori_path)[:-3] + dst_ext
        ori_name, ori_ext = os.path.splitext(os.path.basename(ori_path))
        dst_file = ori_name + dst_ext
        out_path = os.path.join(output_dir, dst_file) #转换后的txt文件存放路径

        txt_obj_list = txt_parse(ori_path, classes=classes)
        obj_xml_write(out_path, txt_obj_list)


def to_yolo_main():
    """(xyxy) 2 (xywh)
        xml 2 txt or yolo  
        txt 2 yolo
    """
    # set
    ori_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/val/label_val"
    output_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/val/label_val_yolo"
    img_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/val/img_val"   # if ext == '.xml', img_dir 可以无
    os.makedirs(output_dir, exist_ok=True)
    ext = '.txt'
    # classes = ["capacitor","resistor", "transistor",
    #             "ic", "pad","inductor","others"]
    classes = ["component"]

    # 
    ori_paths = glob.glob(ori_dir+'/*'+ ext)
    for ori_path in tqdm(ori_paths): #每一张图片都对应一个xml文件这里写xml对应的图片的路径
        print(ori_path)
        yolo_file = os.path.basename(ori_path)[:-3]+'txt'
        out_file = open(os.path.join(output_dir, yolo_file), 'w') #转换后的txt文件存放路径
        
        if ext.find('xml') >= 0:  #xml
            f = open(ori_path)
            xml_text = f.read()
            root = ET.fromstring(xml_text)
            f.close()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
        
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                    float(xmlbox.find('ymax').text))
                # bb = xyxy2xywh((w,h), b)
                bb = [b[0],b[2],b[1],b[3]]
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        else:
            obj_list = txt_parse(ori_path)
            img_path = os.path.join(img_dir, os.path.basename(ori_path).replace('.txt','.jpg'))
            img = cv2.imread(img_path)
            h, w, c = img.shape

            for obj in obj_list:
                cls_id = obj[0]
                b = [obj[1], obj[3], obj[2], obj[4]]
                bb = xyxy2xywh((w,h), b)

                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

        out_file.close()






if __name__ == "__main__":
    to_yolo_main()  
    # json2xml_main()
    # txt2xml_main()