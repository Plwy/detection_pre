from PIL import Image
import shutil
import xml.etree.ElementTree as ET
import xml.dom.minidom
import cv2
import random
import os 

"""
根据xml进行切图，并保存。 图和label 分文件夹保存
"""
def check_xml_export():
    img_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/1"
    ano_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/2"
    output_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/pairs"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    for file in os.listdir(ano_path):
        xml_path = ''
        if file.endswith(".xml"):
            filename = os.path.splitext(os.path.basename(file))[0]
            xml_path = os.path.join(ano_path, file)

        img_path = os.path.join(img_dir, filename+'.jpg')
        print(img_path)
        tree = ET.parse(xml_path)
        img = cv2.imread(img_path)
        print(img.shape)
        for obj in tree.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            print(x_min,x_max,y_min,y_max)
            img_new = img[y_min:y_max,x_min:x_max,:]

            if not os.path.exists(os.path.join(output_path, name)):
                os.makedirs(os.path.join(output_path, name))
                num_ = 1
            else:
                num_ = len(os.listdir(os.path.join(output_path, name))) + 1
            print("==")
            print(os.path.join(output_path, name, name+"_"+str(num_) + ".jpg"))
            
            save_path = os.path.join(output_path, name, name+"_"+str(num_) + ".jpg")

            cv2.imwrite(save_path, img_new)



def check_xml_export2():
    """
    根据xml进行切图，并保存。 图和label 在一个文件夹下
    """
    root_dir = '/media/zsl/data/zsl_datasets/PCB_dataset/pcb_wacv_2019_ext_2'
    output_path = '/media/zsl/data/zsl_datasets/PCB_dataset/pcb_wacv_2019_cls'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    for c_dir, sub_dir, files in os.walk(root_dir):

        if len(files) == 0:
            continue
        else:
            # 先找图
            for file in files:
                if not file.endswith('.jpg') or file.endswith('.png'):
                    continue
                else:
                    filename = file.split('.')[0]
                    img_path = os.path.join(c_dir, file)
                    # 图名找xml
                    xml_path = os.path.join(c_dir, filename+'.xml')
                    if not os.path.isfile(xml_path):
                        continue
                    else:            
                        tree = ET.parse(xml_path)
                        img = cv2.imread(img_path)
                        for obj in tree.findall('object'):
                            name = obj.find('name').text
                            bbox = obj.find('bndbox')
                            x_min = int(bbox.find('xmin').text)
                            y_min = int(bbox.find('ymin').text)
                            x_max = int(bbox.find('xmax').text)
                            y_max = int(bbox.find('ymax').text)
                            print(x_min,x_max,y_min,y_max)
                            img_new = img[y_min:y_max,x_min:x_max,:]

                            if not os.path.exists(os.path.join(output_path, name)):
                                os.makedirs(os.path.join(output_path, name))
                                num_ = 1
                            else:
                                num_ = len(os.listdir(os.path.join(output_path, name))) + 1
                            save_path = os.path.join(output_path, name, name+"_"+str(num_) + ".jpg")

                            # cv2.imwrite(save_path, img_new)
                            try: 
                                print(save_path,img_new.shape)
                                print(xml_path)
                                cv2.imwrite(save_path, img_new)
                            except IOError:
                                print("write error: ",img_path, name)
                            else:
                                continue


def get_pair_crop():
    """
        根据xml框,在对齐了的template 和defect 相同的位置切图
    """
    t_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/1"
    aligned_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/aglined"
    ano_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/2"
    output_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/pairs"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    for file in os.listdir(ano_path):
        xml_path = ''
        if file.endswith(".xml"):
            filename = os.path.splitext(os.path.basename(file))[0]
            xml_path = os.path.join(ano_path, file)

        t_img_path = os.path.join(t_dir, filename+'.jpg')
        t_img = cv2.imread(t_img_path)
        print(t_img_path)
        print(t_img.shape)

        d_img_path = os.path.join(aligned_dir, filename+'_aligned_1.jpg')
        d_img = cv2.imread(d_img_path)

        tree = ET.parse(xml_path)
        for obj in tree.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            print(x_min,x_max,y_min,y_max)
            t_img_crop = t_img[y_min:y_max,x_min:x_max,:]

            d_img_crop = d_img[y_min:y_max,x_min:x_max,:]
            # img_new = img[x_min:x_max,y_min:y_max,:]


            # cv2.imshow("img_new",img_new)
            # cv2.waitKey(0)

            if not os.path.exists(os.path.join(output_path, name)):
                os.makedirs(os.path.join(output_path, name))
                num_ = 1
            else:
                num_ = len(os.listdir(os.path.join(output_path, name))) + 1
            print("==")
            print(os.path.join(output_path, name, name+"_"+str(num_) + ".jpg"))
            
            save_t_path = os.path.join(output_path, name, name+"_t_"+str(num_) + ".jpg")
            save_d_path = os.path.join(output_path, name, name+"_d_"+str(num_) + ".jpg")


            cv2.imwrite(save_t_path, t_img_crop)
            cv2.imwrite(save_d_path, d_img_crop)




def get_pair_crop2():
    """
        在对齐了的template 和defect 相同的位置切图, 保存在/t/, /d/文件夹子
    """
    t_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/1"
    aligned_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/aglined"
    ano_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/2"
    output_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/all_pairs"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    aligned_list = os.listdir(aligned_dir)

    
    for file in os.listdir(ano_path):
        xml_path = ''
        if file.endswith(".xml"):
            filename = os.path.splitext(os.path.basename(file))[0]
            xml_path = os.path.join(ano_path, file)

        t_img_path = os.path.join(t_dir, filename+'.jpg')
        t_img = cv2.imread(t_img_path)
        print(t_img_path)
        print(t_img.shape)

        # 找到该模板对应的缺陷图们
        for aligned_file in aligned_list:
            if aligned_file.find(filename) == 0: 
                d_filename = os.path.splitext(os.path.basename(aligned_file))[0]


                # d_img_path = os.path.join(aligned_dir, filename+'_aligned_1.jpg')
                d_img_path = os.path.join(aligned_dir, aligned_file)
                d_img = cv2.imread(d_img_path)

                tree = ET.parse(xml_path)
                for obj in tree.findall('object'):
                    name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    x_min = int(bbox.find('xmin').text)
                    y_min = int(bbox.find('ymin').text)
                    x_max = int(bbox.find('xmax').text)
                    y_max = int(bbox.find('ymax').text)
                    # print(x_min,x_max,y_min,y_max)
                    t_img_crop = t_img[y_min:y_max,x_min:x_max,:]

                    d_img_crop = d_img[y_min:y_max,x_min:x_max,:]
                    # img_new = img[x_min:x_max,y_min:y_max,:]



                    # cv2.imshow("img_new",img_new)
                    # cv2.waitKey(0)

                    output_t_dir = os.path.join(output_path, "t")
                    if not os.path.exists(output_t_dir):
                        os.makedirs(output_t_dir)
                        num_ = 1
                    else:
                        num_ = len(os.listdir(output_t_dir)) + 1

                    output_d_dir = os.path.join(output_path, "d")
                    if not os.path.exists(output_d_dir):
                        os.makedirs(output_d_dir)
                        num_ = 1
                    else:
                        num_ = len(os.listdir(output_d_dir)) + 1


                    # print(os.path.join(output_t_dir, filename+"_"+str(num_) + ".jpg"))
                    
                    save_t_path = os.path.join(output_t_dir, filename + "_"+str(num_) + ".jpg")
                    save_d_path = os.path.join(output_d_dir, d_filename+"_"+str(num_) + ".jpg")


                    cv2.imwrite(save_t_path, t_img_crop)
                    cv2.imwrite(save_d_path, d_img_crop)
            else:
                continue



if __name__ == '__main__':
    get_pair_crop()
