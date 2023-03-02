import shutil
import xml.etree.ElementTree as ET
import cv2
import os
import glob
from tqdm import tqdm
import codecs

from parse_objfile import txt_parse
 

def convert_reverse(size, box):
    '''Back out pixel coords from yolo format
    input = image_size (w,h), 
        box = [x,y,w,h]'''
    x,y,w,h = box
    dw = 1./size[0]
    dh = 1./size[1]
    
    w0 = w/dw
    h0 = h/dh
    xmid = x/dw
    ymid = y/dh
    
    x0, x1 = xmid - w0/2., xmid + w0/2.
    y0, y1 = ymid - h0/2., ymid + h0/2.

    return [x0, x1, y0, y1]

def convert2yolo(size, box):
    """convert 2 yolo format
        xyxy 2 xywh 
    """
 
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


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


def to_yolo_main():
    """(xyxy) 2 (xywh)
        xml 2 txt or yolo  
        txt 2 yolo
    """
    # set
    ori_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/base_crops_new/img_crop_labels"
    output_dir = "/media/zsl/data/zsl_datasets/for_train/717_base/base_crops_new/img_crop_labels_txt"
    img_dir = "/media/zsl/data/zsl_datasets/717/basedata/all/imgs"   # if ext == '.xml', img_dir 可以无
    os.makedirs(output_dir, exist_ok=True)
    ext = '.xml'
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
                # bb = convert2yolo((w,h), b)
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
                bb = convert2yolo((w,h), b)

                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

        out_file.close()

def txt2xml_main():
    """
        读取当前目录的文件,转化为其他格式,或进行更改
        txt format: [cls_id, x_min,y_min,x_max,y_max]
    """
    # 原始文档目录
    ori_dir = "/media/zsl/data/zsl_datasets/717/pro/test_lab/result2w_all_t/exp/labels"
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

 
if __name__ == '__main__':
    to_yolo_main()
