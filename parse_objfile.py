import os
import glob
import cv2
import shutil
from tqdm import tqdm
import json
import xml.etree.ElementTree as ET

def json_parse(json_path):
    """"
        output: labels = [[x_min,y_min,x_max,y_max,label],...]
    """
    labels=[]
    with open(json_path, 'r') as f:
        # 解析json
        info = json.load(f)
        roi_list = info["shapes"]
        h = info["imageHeight"]
        w = info["imageWidth"]

        for roi in roi_list:
            label = roi["label"]
            # 读json中的roi点， 左下点，右上点
            point = roi["points"]
            x1,y1= point[0]
            x2,y2 = point[1]
            x_min = float(min(x1,x2))
            x_max = float(max(x1,x2))
            y_min = float(min(y1,y2))
            y_max = float(max(y1,y2))
            b = [x_min,y_min,x_max, y_max, label]
            labels.append(b)

    return labels, h, w

def txt_parse(txt_path, classes=None):
    r = []
    with open(txt_path, 'r') as f:  
        a = f.readlines()
        for l in a:
            l = l.strip()
            x = l.split(' ')
            if classes is not None:
                cls = classes[int(x[0])]  # 传名称
            else:
                cls = x[0] # 传类别号
            ll =(cls, float(x[1]), float(x[2]), float(x[3]), float(x[4]))
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
