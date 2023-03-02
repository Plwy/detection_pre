import os 
import cv2
import shutil
import glob
import xml.etree.ElementTree as ET
import xml.dom.minidom
import sys
"""
    遍历公开数据集的xml文件,去除和,更改其中的标签
"""
def remove_element(xml_path):
    tr = ET.parse(xml_path)
    for element in tr.iter():
        for subElement in element:
            if subElement.tag == "path":
                # se = subElement.get("path")
                element.remove(subElement)
    tr.write(sys.stdout)



def pro_main():
    """
    # get  all xml files in root
    改标签,删标签
    去除text, component text 去丝印等
    """

    root_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/pcb_wacv_2019_ext_2"
    
    xml_paths = []
    for c_dir, sub_dir, files in os.walk(root_dir):
        if len(files) == 0:
            continue
        else:
            for file in files:
                xml_path = os.path.join(c_dir, file)
                if file.endswith('_new.xml'):
                    cm = 'rm ' + xml_path
                    os.system(cm)
                    continue
                if file.endswith('.xml'):
                    xml_paths.append(xml_path)

    print('process file:',xml_paths)


    # pro xml files
    c_dic = {'text':0,'comp_text':0, 'containor':0, 'pad':0, 'resistor':0}
    xml_news = []
    for path_xml in xml_paths:
        c_c = 0

        tree = ET.parse(path_xml)
        root = tree.getroot()
        for child in root.findall('object'):
            name = child.find('name').text
            # print(name)
            if name.find('text') == 0:
                root.remove(child)

            elif name.find('component text') >= 0:
                c_c +=1 
                root.remove(child)

            elif name.find('pins') >= 0:
                root.remove(child)

            elif name.find('unknown') == 0:
                child.find('name').text = 'ferrite bead'

            # 修改名字
            elif name.find('resistor') >= 0:
                child.find('name').text = 'resistor'

            elif name.find('pad') >= 0:
                child.find('name').text = 'pad'

            elif name.find('capacitor') >= 0:
                child.find('name').text = 'capacitor'

            elif name.find('connector') >= 0:
                child.find('name').text = 'connector'

            elif name.find('ic') >= 0:
                child.find('name').text = 'ic'

            elif name.find('transistor') >= 0:
                child.find('name').text = 'transistor'

            elif name.find('diode') >= 0:
                child.find('name').text = 'diode'


            elif name.find('led') >= 0:
                child.find('name').text = 'led'

            elif name.find('switch') >= 0:
                child.find('name').text = 'switch'

            elif name.find('clock') >= 0:
                child.find('name').text = 'clock'

            elif name.find('test point') >= 0:
                child.find('name').text = 'test point'

            elif name.find('emi filter') >= 0:
                child.find('name').text = 'emi filter'

            elif name.find('electrolytic capacitor') >= 0:
                child.find('name').text = 'electrolytic capacitor'

            elif name.find('jumper') >= 0:
                child.find('name').text = 'jumper'

            elif name.find('button') >= 0:
                child.find('name').text = 'button'

            elif name.find('inductor') >= 0:
                child.find('name').text = 'inductor'

            elif name.find('ferrite bead') >= 0:
                child.find('name').text = 'ferrite bead'

            else:
                child.find('name').text = name.split(' ')[0]
        print(c_c)
        xml_new = path_xml.replace('.xml', '_new.xml')
        xml_news.append(xml_new)
        tree.write(xml_new)

    # check
    dic = {}
    for path_xml in xml_news:
        tree = ET.parse(path_xml)
        root = tree.getroot()
        for child in root.findall('object'):
            name = child.find('name').text  
            if name not in dic:
                dic[name] = 1
            else:
                dic[name] += 1

            if name.find('text') == 0:
                print("wrong text")
            elif name.find('component text') >= 0:
                print("wrong  component text")

            elif name.find('resistor') == 0:
                if len(name) > len('resistor'):
                    print("wrong  resistor")

            elif name.find('pad') == 0:
                if len(name) > len('pad'):
                    print("wrong  pad")

            elif name.find('capacitor') == 0:
                if len(name) > len('capacitor'):
                    print("wrong  capacitor")

            elif name.find('connector') == 0:
                if len(name) > len('connector'):
                    print("wrong  connector")

            elif name.find('ic') == 0:
                if len(name) > len('ic'):
                    print("wrong  ic")

            elif name.find('transistor') == 0:
                if len(name) > len('transistor'):
                    print("wrong  transistor")

            elif name.find('diode') == 0:
                if len(name) > len('diode'):
                    print("wrong  diode")

            elif name.find('led') == 0:
                if len(name) > len('led'):
                    print("wrong  led")

            elif name.find('switch') == 0:
                if len(name) > len('switch'):
                    print("wrong  switch")

            elif name.find('clock') == 0:
                if len(name) > len('clock'):
                    print("wrong  clock")

    # 打印清理最后的类别及数量
    for k in dic:
        print(k, ':', dic[k])


    for c_dir, sub_dir, files in os.walk(root_dir):
        if len(files) == 0:
            continue
        else:
            for file in files:
                xml_path = os.path.join(c_dir, file)
                if file.endswith('_new.xml'):        
                    continue
                if file.endswith('.xml'):
                    cm = 'rm ' + xml_path
                    os.system(cm)



def pro_main2():
    """
        scheme1, 清理pcb_wav 和717的标签类别
    """

    # root_dir = "/media/zsl/data/zsl_datasets/for_train/scheme1/all_train"
    # root_dir = "/media/zsl/data/zsl_datasets/717/labeled/component_label/templates_nobound2/xml"
    root_dir = "/media/zsl/data/zsl_datasets/for_train/scheme1/pcba_wav_base/all_train3_crop/aug_labels_scl"
    # cls_scheme1 = ["resistor", "inductor", "capacitor", "transistor",
    #                 "ic", "pad", "others"]

    # cls_scheme1 = ["resistor", "inductor", "capacitor", "transistor",
    #                 "ic", "others"]

    cls_scheme1 = ["component"]


    xml_paths = []
    for c_dir, sub_dir, files in os.walk(root_dir):
        if len(files) == 0:
            continue
        else:
            ## 先删除已有的新生成文件
            for file in files:
                xml_path = os.path.join(c_dir, file)
                if file.endswith('_new.xml'):
                    cm = 'rm ' + xml_path
                    os.system(cm)
                    continue
                if file.endswith('.xml'):
                    xml_paths.append(xml_path)

    print('process file:',len(xml_paths))


    # 对xml 标签进行 合并,删除和修改
    xml_news = []
    c_c = 0
    for path_xml in xml_paths:

        tree = ET.parse(path_xml)
        root = tree.getroot()
        for child in root.findall('object'):

            name = child.find('name').text
            # if name.find('diode') == 0 or name.find('tirode') == 0:
            #     c_c += 1
            #     child.find('name').text = 'transistor'
            #     name = child.find('name').text
            
            # # # delete pad cls
            # # if name.find('pad') == 0:
            # #     root.remove(child)

            # if name not in cls_scheme1:
            #     child.find('name').text = 'other'
            # else:
            #     continue

            if name.find('pad') == 0:
                root.remove(child)
            if name not in cls_scheme1:
                child.find('name').text = cls_scheme1[0]
            

                
        xml_new = path_xml.replace('.xml', '_new.xml')
        xml_news.append(xml_new)
        tree.write(xml_new)

    # print("diode ande tirode nums:",c_c)

    # check
    dic = {}
    for path_xml in xml_news:
        tree = ET.parse(path_xml)
        root = tree.getroot()
        for child in root.findall('object'):
            name = child.find('name').text  
            if name not in dic:
                dic[name] = 1
            else:
                dic[name] += 1

            # if name not in cls_scheme1:
            #     print("wrong  label:",name)

    # 打印清理最后的类别及数量
    for k in dic:
        print(k, ':', dic[k])

    for c_dir, sub_dir, files in os.walk(root_dir):
        if len(files) == 0:
            continue
        else:
            for file in files:
                xml_path = os.path.join(c_dir, file)
                if file.endswith('_new.xml'):        
                    continue
                if file.endswith('.xml'):
                    cm = 'rm ' + xml_path
                    os.system(cm)


def rename():
    """
        _new.xml 改为 .xml
    """
    root_dir = '/media/zsl/data/zsl_datasets/for_train/scheme1/pcba_wav_base/all_train3_crop/aug_labels_scl'
    c = 0
    for c_dir, sub_dir, files in os.walk(root_dir):
        if len(files) == 0:
            continue
        else:
            for file in files:
                xml_path = os.path.join(c_dir, file)
                if file.endswith('_new.xml'):  
                    c += 1
                    xml_path2 = xml_path.replace('_new.xml', '.xml')
                    cm = 'mv ' + xml_path + ' ' + xml_path2
                    os.system(cm)

    print("rename :",c)

if __name__ == "__main__":
    pro_main2()
    rename()