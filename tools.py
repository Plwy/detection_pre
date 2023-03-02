import glob
import os
import shutil
def mv_matchfile():
    match_strs = ["少件2022050901013_aligned",
                    "少件2022050901014_aligned",
                    "少件2022050901015_aligned",
                    "少件2022050901016_aligned",
                    "少件2022050901017_aligned",
                    "少件2022050901018_aligned",
                    "少件2022050901019_aligned"]
    dir = "/home/code/zsl_datasets/all_train3/labels/val"
    dst_dir = "/home/code/zsl_datasets/wrong_label"
    file_list = glob.glob(dir+'/*')
    print(len(file_list))
    c = 0
    for file in file_list:
        file_name = os.path.basename(file)
        for s in match_strs:
            if file_name.find(s) < 0:
                continue
            else:
                # print(file_name)
                c += 1
                dst = os.path.join(dst_dir, file_name)
                shutil.move(file, dst)
    print("match file num:",c)


def add_withor():
    """
        每个切图文件中添加原图
    """
    dst = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_crop_withori_labels_6wbig_v729"
    src = "/media/zsl/data/zsl_datasets/PCB_test/HR_test2/compare_new/defects_aligned_white_labels_6wbig_v722"

    x = 0
    for cur, subs, files in os.walk(src):
        if len(files)==0:
            continue
        else:
            f_2, f_1 = cur.split('/')[-2], cur.split('/')[-1]
            print(f_2, f_1)
            for file in files:
                file_name, ext = os.path.splitext(os.path.basename(file))
                src_path = os.path.join(cur, file)
                dst_path = os.path.join(dst, f_1, file_name, file)
                print(src_path)
                print(dst_path)

                # shutil.copy(src_path, dst_path)
                x += 1
                break
    print(x)




if __name__ == '__main__':
    mv_matchfile()
    add_withor()