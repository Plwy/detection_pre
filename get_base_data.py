import os
import shutil
import cv2
from tqdm import tqdm 
"""
    筛选用作增强的基础图

"""

def get_base_main():
    img_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset/PCB_data_all"
    output_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/PCB-dataset/PCB_data_base"

    img_lists = os.listdir(img_dir)

    base_list = []
    n = 100
    c = 0
    for img_name in tqdm(img_lists):
        # if c > n:
        #     break
        filename = os.path.splitext(os.path.basename(img_name))[0]
        if filename.find("copyMin") < 0 and filename.find("Flip") < 0 and filename.find("Hsv") < 0 and filename.find("HSV") < 0:
            print(filename)

            save_path = os.path.join(output_dir,img_name)
            # cv2.imwrite(save_path, img_name)
            src_path = os.path.join(img_dir, img_name)
            target_path = os.path.join(output_dir, img_name)
            shutil.copy(src_path, target_path)
            c += 1

if __name__ == "__main__":
    get_base_main()