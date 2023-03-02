from genericpath import exists
import os
import glob
from posixpath import basename
import random
import cv2
import shutil
from tqdm import tqdm

"""
    构造数据集的一些常用函数
"""

def data_split():
    """
    文件分发，一份文件夹分几份图
    eg：如文件名为a，切分文件a_1, a_2,...a_n存储
    """
    image_root = '/media/zsl/data/2021make_dataset/datasets/GT1/all/c_ori_ocrfalse'
    save_root = '/media/zsl/data/2021make_dataset/datasets/GT1/datateam_small'
    split = 1/4

    os.makedirs(save_root, exist_ok=True)
    name_list = os.listdir(image_root)
    print(len(name_list))

    part_num = split * len(name_list)

    image_root_name = os.path.basename(image_root)

    part_index = 0
    part = 0
    for i, image_name in enumerate(tqdm(name_list)):
        if i >= part:
            part_index += 1
            part += part_num
        image_path = os.path.join(image_root, image_name)

        image = cv2.imread(image_path)
        save_fold_name = image_root_name + str(part_index)
        save_fold = os.path.join(save_root, save_fold_name)
        if not os.path.exists(save_fold):
            os.makedirs(save_fold, exist_ok=True)

        save_path = os.path.join(save_root, save_fold_name, image_name)
        shutil.copy(image_path, save_path)

def check_split_merge():
    """
        分发检查。检查几个子文件夹，合起来是不是与父文件夹一致
    """
    split_dir1 = '/media/zsl/data/2021make_dataset/datasets/GT0/a_kqh'
    split_dir2 = '/media/zsl/data/2021make_dataset/datasets/GT0/a_sdd'
    split_dir3 = '/media/zsl/data/2021make_dataset/datasets/GT0/a_szt'
    split_dir4 = '/media/zsl/data/2021make_dataset/datasets/GT0/a_ws'

    father_dir = '/media/zsl/data/2021make_dataset/datasets/GT0/a'
    merge = []

    split1 = os.listdir(split_dir1)
    split2 = os.listdir(split_dir2)
    split3 = os.listdir(split_dir3)
    split4 = os.listdir(split_dir4)
    merge.extend(split1)
    merge.extend(split2)
    merge.extend(split3)
    merge.extend(split4)
    
    father = os.listdir(father_dir)
    y = [name for name in father if name in merge] # 筛选相同的文件

    print(y[:5])
    print(len(y))

def record_dirname():
    """
        将文件夹内的文件名写入txt
    """
    file_path = glob("/media/zsl/data/make_dataset/image1_part2/image1_part2/*")
    file_path2 = glob("/media/zsl/data/make_dataset/image4_part2/image4_part2/*")

    print(len(file_path),len(file_path2))
    file_path.extend(file_path2)
    with open('./zsl_new.txt','w') as f:
        for i in file_path:
            text = i.split("/")[-1]
            f.write(text + "\n")
    
def get_image_without_gt():
    """
        图像文件找对应的gt text
        没有对应的text的图像文件名写入表
    """
    img_dir = '/media/zsl/data/zsl_datasets/GF/data_set/train_txtimg_54909/train_54360/hr_image'
    t_path = '/media/zsl/data/zsl_datasets/GF/gaofa_large_20w/text'
    save = 'dif-train-54360-hr.csv'

    img_list = os.listdir(img_dir)
    text_list = os.listdir(t_path)
    dif = []
    count = 0
    for name in img_list:
        if not name.endswith('.png'):
            dif.append(name)

        name_x = name.replace('.png','.txt')
        if name_x in text_list:
            continue
        else:
            dif.append(name)

    print(len(dif))
    with open(save, 'w') as f:
        f.writelines(dif)

def get_some4train():
    """
    数据集中选出一些图，合适尺寸，用作非成对的训练
    """
    gf_5w = '/media/zsl/data/2021make_dataset/datasets/GT1/all/c_ori_ocrfalse'
    save_path = '/media/zsl/data/2021make_dataset/Test/merge_ocr_test/position_test/ori_false'
    os.makedirs(save_path, exist_ok=True)

    select_n = 1000

    name_list = os.listdir(gf_5w)
    print(len(name_list))
    name_list = random.sample(name_list, len(name_list))

    count = 0
    for image_name in name_list:
        if count >= select_n:
            break
        image_path = os.path.join(gf_5w, image_name)

        image = cv2.imread(image_path)
        # h,w,c = image.shape
        # print(h, w)
        shutil.copy(image_path, save_path)
        count += 1
        # # min_b = min(h,w)
        # if min_b >= 384:
        #     shutil.copy(image_path, save_path)
        #     # target_path = os.path.join(save_path, image_name)
        #     # command = 'cp ' + image_path + ' ' + target_path
        #     # os.system(command) 
        #     # print(command)
        #     count += 1
        # else:
        #     continue
    print(count)

def rm_lr_nohr():
    """
        df2k中筛选出成对的lr hr
    """
    dir_hr = '/media/zsl/data/zsl_datasets/GF/data_set/DF2K300dpi/DF2K_HR'
    dir_lr = '/media/zsl/data/zsl_datasets/GF/data_set/DF2K300dpi/DF2K_LR_bicubic/X2'
    dir_lr_withhr = '/media/zsl/data/zsl_datasets/GF/data_set/DF2K300dpi/DF2K_LR_bicubic/X2_lr'

    scale = [2]
    ext = ['.png','.png']

    names_hrs =[os.path.basename(file) for file in sorted(
        glob.glob(os.path.join(dir_hr, '*' + ext[0]))
    )]
    names_lrs = sorted(
        glob.glob(os.path.join(dir_lr, '*' + ext[1]))
    )

    names_hr = []
    names_lr = [[] for _ in scale]
    pair = 0
    for f in names_lrs:
        filename= os.path.basename(f)
        for si, s in enumerate(scale):
            names_lr[si].append(f)
        filename_hr = filename.replace('x2','')
        if filename_hr in names_hrs:
            current_path = os.path.join(dir_lr, filename)
            target_path =  os.path.join(dir_lr_withhr, filename) 
            command = 'cp '+ current_path + ' ' + target_path
            os.system(command)
            print('pair !')
            pair += 1
        else:
            break
    print(pair)
    return

def sr_resize_save():
    """
    sr 图 全部resize 到其1/2尺寸 然后保存

    """
    image_dir = '/media/zsl/data/workspace/codes/DRN/experiment/test/results/gf3w_result_model467/x2'
    save_dir = '/media/zsl/data/workspace/codes/DRN/experiment/test/results/gf3w_result_model467/x2_r'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_list = os.listdir(image_dir)
    i = 0
    for filename in file_list:
        img_path = os.path.join(image_dir, filename)
        src_img = cv2.imread(img_path)

        size = 0.5
        src_img = cv2.resize(src_img, (0, 0), fx=size, fy=size)
        save_img_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_img_path , src_img)
        i += 1
    print(i)
    print(len(file_list))

def png_tran_jpg():
    """
    png图转jpg

    """
    image_dir = '/media/zsl/data/workspace/codes/DRN/experiment_imp/test/results/test267_result_model57_detect/x2'
    save_dir = '/media/zsl/data/workspace/codes/DRN/experiment_imp/test/results/test267_result_model57_detect/x2_format'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_list = os.listdir(image_dir)
    i = 0
    for filename in file_list:
        img_path = os.path.join(image_dir, filename)
        src_img = cv2.imread(img_path)

        name_pre = os.path.splitext(filename)[0]
        name_suf = '.jpg'

        save_img_path = os.path.join(save_dir, name_pre+name_suf)
        
        cv2.imwrite(save_img_path , src_img)
        i += 1
    print(i)
    print(len(file_list))

def assign_data(data_dir, save_path, select_n):
    """
        随机分发数据 for szt
    """
    os.makedirs(save_path, exist_ok=True)
    name_list = os.listdir(data_dir)
    print('文件数目:',len(name_list))
    random_select_list = random.sample(name_list, select_n)

    count = 0
    for image_name in random_select_list:
        image_path = os.path.join(data_dir, image_name) 
        # cp
        # shutil.copy(image_path, save_path)
        # mv
        target_save_path = os.path.join(save_path, image_name)
        shutil.move(image_path, target_save_path)

        count += 1

    print("随机筛选文件数:",count)
    print("移动文件至路径:", save_path)

def assign_data_main():
    # 大的数据集的路径
    data_dir = '/media/zsl/data/zsl_datasets/for_train/scheme1/train_debug/images/train'
    # 随机剪切出来的存储路径
    save_path = '/media/zsl/data/zsl_datasets/for_train/scheme1/train_debug/images/val'
    # 随机筛选的数目
    select_n = 500
    assign_data(data_dir, save_path, select_n)

def get_pair():
    """
        将两个文件夹的相同名文件分别拷贝到目标文件夹
    """
    # 文件夹1
    image_path = '/media/zsl/data/zsl_datasets/GF/gaofa_large_20w/HR/image_137376'
    # 文件夹2
    gt_path = '/media/zsl/data/zsl_datasets/GF/gaofa_large_20w/text'
    # 文件夹1中的配对文件存储路径
    target_path = './HR440/'
    # 文件夹2中的配对文件存储路径
    target_gt_path = './HR440_gt/'

    os.makedirs(target_path, exist_ok=True)
    os.makedirs(target_gt_path, exist_ok=True)

    # 遍历
    image_list = os.listdir(image_path)
    image_ext = os.path.splitext(image_list[0])[-1]
    x = [os.path.splitext(name)[0] for name in image_list ]

    gt_list = os.listdir(gt_path)
    gt_ext = os.path.splitext(gt_list[0])[-1]
    y = [os.path.splitext(name)[0] for name in gt_list ]
    print("file_dir 1 lens:", len(x))
    print(x[:5])
    print("file_dir 2 lens:", len(y))
    print(y[:5])

    # 配对
    zshare = [i for i in x if i in y ]
    print('same name count:',len(zshare))

    count = 0
    for name in zshare:
        print(name)
        imgfile = os.path.join(image_path, name + image_ext)
        gtfile = os.path.join(gt_path, name + gt_ext)

        shutil.copy(imgfile, target_path)
        shutil.copy(gtfile, target_gt_path)
        count += 1

    print('copy file total:',count)

def mv_diff():
    """
        移除无groundtruth的文件
        即当前文件夹内，与目标文件夹找不到同名的文件。移出到targe文件夹子内
    """
    # 文件夹
    image_path = '/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/1_big_obj_crops_labels_augs'
    # 目标文件夹
    gt_path = '/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/1_big_obj_crops_imgs_augs'
    # 移出目标文件夹中的无配对文件存储路径
    target_path = '/media/zsl/data/zsl_datasets/for_train/717_base/big_obj_center_crop/llllll'

    os.makedirs(target_path, exist_ok=True)

    # 遍历
    image_list = os.listdir(image_path)
    image_ext = os.path.splitext(image_list[0])[-1]
    src = [os.path.splitext(name)[0] for name in image_list ]
    gt_list = os.listdir(gt_path)
    gt_ext = os.path.splitext(gt_list[0])[-1]
    target = [os.path.splitext(name)[0] for name in gt_list ]
    print("file_dir 1 lens:", len(src))
    print(src[:5])
    print("file_dir 2 lens:", len(target))
    print(target[:5])

    # 查找不成对
    zdiff = [i for i in src if i not in target ]
    print('diff name count',len(zdiff))

    count = 0
    for name in zdiff:
        imgfile = os.path.join(image_path, name + image_ext)
        if os.path.isfile(imgfile):
            target_image_path = os.path.join(target_path, name + image_ext)
            # command = 'mv '+ imgfile + ' ' + target_image_path
            # print(command)
            # os.system(command)
            shutil.move(imgfile, target_path)

            # shutil.copy(imgfile, target_path)
            count += 1
        else:
            print('src without file:', imgfile)
            continue

    # print('move file total:',count)

def mv_same():
    """
        从目标文件夹内，找到与当前文件夹同名的文件。移出到或拷贝到targe文件夹子内
    """
    # 当前文件夹
    image_path = '/media/zsl/data/zsl_datasets/for_train/scheme1/train_debug/images/val'
    # 目标文件夹
    gt_path = '/media/zsl/data/zsl_datasets/for_train/scheme1/train_debug/labels/train'
    # 移出路径
    target_path = '/media/zsl/data/zsl_datasets/for_train/scheme1/train_debug/labels/val'
    os.makedirs(target_path, exist_ok=True)

    # 遍历
    image_list = os.listdir(image_path)
    image_ext = os.path.splitext(image_list[0])[-1]

    x = [os.path.splitext(name)[0] for name in image_list ]

    gt_list = os.listdir(gt_path)
    gt_ext = os.path.splitext(gt_list[0])[-1]
    # y = [os.path.splitext(name)[0].replace('x2','') for name in gt_list ]
    y = [os.path.splitext(name)[0] for name in gt_list ]

    print("file_dir 1 lens:", len(x))
    print(x[:5])
    print("file_dir 2 lens:", len(y))
    print(y[:5])

    # 查找成对
    zsame = [i for i in x if i in y ]
    print('same name count',len(zsame))

    count = 0
    for name in zsame:
        imgfile = os.path.join(gt_path, name + gt_ext)
        if os.path.isfile(imgfile):
            target_image_path = os.path.join(target_path, name + gt_ext)
            # shutil.move(imgfile, target_image_path)

            shutil.copy(imgfile, target_image_path)
            count += 1
        else:
            print('src without file:', imgfile)
            continue

    print('op file total:',count)

def _random_crop_pair_img(lr, hr, patch_size=(384,384), times=1, scale=2):
    lr_h, lr_w = lr.shape[:2]
    print(lr_h, lr_w)
    hr_wp = patch_size[0]
    hr_hp = patch_size[1]

    lr_wp = hr_wp // scale
    lr_hp = hr_hp // scale

    lr_patch_list = []
    hr_patch_list = []
    for i in range(times):
        # 随机得到lr切图的左上角点
        lr_px = random.randrange(0, lr_w - lr_wp + 1)
        lr_py = random.randrange(0, lr_h - lr_hp + 1)
        hr_px, hr_py = scale * lr_px, scale * lr_py
        print('x:{}-{},y:{}-{}'.format(lr_px,lr_px+lr_wp, lr_py,lr_py+lr_hp))

        lr_patch = lr[lr_py:lr_py+lr_hp, lr_px:lr_px+lr_wp, :]
        hr_patch = hr[hr_py:hr_py+hr_hp, hr_px:hr_px+hr_wp, :]    

        lr_patch_list.append(lr_patch)
        hr_patch_list.append(hr_patch)
    return lr_patch_list, hr_patch_list

def _random_crop_img(img_lr_path, img_hr_path, lr_save_dir, hr_save_dir, patch_size=(384,384), times=1):
    lr = cv2.imread(img_lr_path)
    hr = cv2.imread(img_hr_path)

    lr_patch_list, hr_patch_list = _random_crop_pair_img(lr, hr, patch_size, times)

    lr_image_name = os.path.basename(img_lr_path)
    hr_image_name = os.path.basename(img_hr_path)
    lr_name_split = os.path.splitext(lr_image_name)
    hr_name_split = os.path.splitext(hr_image_name)
    lr_name_pre, lr_name_ext = lr_name_split[0], lr_name_split[-1]
    hr_name_pre, hr_name_ext = hr_name_split[0], hr_name_split[-1]

    for i in range(len(lr_patch_list)):
        lr_new_name = lr_name_pre+'_patch'+str(i)+lr_name_ext
        hr_new_name = hr_name_pre+'_patch'+str(i)+hr_name_ext
        print(lr_new_name)
        print(hr_new_name)

        lr_patch_save = os.path.join(lr_save_dir, lr_new_name)
        hr_patch_save = os.path.join(hr_save_dir, hr_new_name)
        print(lr_patch_save)
        print(hr_patch_save)

        cv2.imwrite(lr_patch_save, lr_patch_list[i])
        cv2.imwrite(hr_patch_save, hr_patch_list[i])

def random_crop_img_main():
    """
        从成对大图中切出成对的小图
        先对图进行n次随机切图，切图到128×32小图， 256x64大图
    """
    hr_dir = '/media/zsl/data/zsl_datasets/GF/data_set/DF2K300dpi/DF2K_HR_train'
    lr_dir = '/media/zsl/data/zsl_datasets/GF/data_set/DF2K300dpi/DF2K_LR_bicubic/X2_lr_train_5162'
    hr_save_dir = '/media/zsl/data/zsl_datasets/GF/data_set/300dpi_crop/HR_crop'
    lr_save_dir = '/media/zsl/data/zsl_datasets/GF/data_set/300dpi_crop/LR_crop'

    os.makedirs(lr_save_dir ,exist_ok=True)
    os.makedirs(hr_save_dir ,exist_ok=True)
    crop_times = 10  # 每张图随机切的次数
    patch_size = (384,384)
    # 
    img_lr_name_list = os.listdir(lr_dir)
    for i, lr_name in enumerate(tqdm(img_lr_name_list)):
        img_lr_path = os.path.join(lr_dir,lr_name)

        hr_name = lr_name.replace('x2', '')
        img_hr_path = os.path.join(hr_dir, hr_name)

        _random_crop_img(img_lr_path, img_hr_path, lr_save_dir, hr_save_dir, patch_size, crop_times)

def mv_txt_same():
    """
        按照txt文件中名字，移除对应文件
    """
    txt_path = '/media/zsl/data/make_dataset/zsl_del.txt'
    file_path = '/media/zsl/data/make_dataset/div_zsl/c'
    mv_path = '/media/zsl/data/make_dataset/div_zsl/del/c'


    if not os.path.exists(mv_path): os.mkdir(mv_path)

    file_list = os.listdir(file_path)
    print('before total num:', len(file_list))

    same = []
    with open(txt_path, 'r') as f:
        target = f.readlines()
        for line in target:
            line = line.strip('\n')

            if line in file_list:
                same.append(line)
            else:
                continue

    for i, name in enumerate(tqdm(same)):
        samefile = os.path.join(file_path, name)
        file_mv_path = os.path.join(mv_path, name)
        command = 'mv '+ samefile + ' ' + file_mv_path
        # print(command)
        os.system(command)        

    print('after total num:', len(os.listdir(file_path)))
    print('mv path num:', len(os.listdir(mv_path)))

def auto_split_change():
    """
        将提交的修改图，替换文件夹下的已有图
    """
    img_new_dir = "/media/zsl/data/2021make_dataset/Test/auto_split_mod_test/20210326_szt_修改"
    img_old_dir = "/media/zsl/data/2021make_dataset/Test/auto_split_mod_test/20210325_szt_b_cat"
    img_new_list = os.listdir(img_new_dir)

    def tral_dir(base_dir):
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                file_path = os.path.join(root, f)
                yield file_path

    # 对每张改过的图
    count = 0
    for file_name in img_new_list:
        img_new_path = os.path.join(img_new_dir, file_name)
        flag = False   # 标记是否找到对应文件

        # 遍历提交的文件夹
        for file_path in tral_dir(img_old_dir):
            old_file_name = os.path.basename(file_path)
            if old_file_name == file_name:
                shutil.copy(img_new_path, file_path)   
                flag = True
                count += 1
                break       # 提交文件重复，则需要将其都替换，则不加break

        if flag:
            continue
        else:
            print("can find {} in commit files, so copy failed".format(file_name))

    print("change file nums:", count)
    
def rename_files():
    file_dst = '/media/zsl/data/workspace/LLIE_proj/datasets/dataset_0118_data4/test/paired_data_rename'

    file_list = os.listdir(file_dst)

    s1 = '_DRN_seg_bin' # ori
    s2 = '' # target

    for i, file_name in enumerate(file_list):
        s1 = file_name.split('.')[-2]
        s2 = file_name.split('.')[-2].split('_')[-1]
        file_name_new = file_name.replace(s1, s2)
        
        old_path = os.path.join(file_dst, file_name)
        new_path = os.path.join(file_dst, file_name_new)

        os.rename(old_path , new_path)

def check_file_exist():
    target_dir = '/home/zsl/Downloads/三个数据集精修任务/三个数据集精修任务/整合已有数据文件'
    check_file_name = '30428100_4_48.png'

    def tral_dir(base_dir):
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                file_path = os.path.join(root, f)
                yield file_path

    # 遍历提交的文件夹
    for file_path in tral_dir(target_dir):
        file_name = os.path.basename(file_path)
        if file_name == check_file_name:
            print("exist:")
            print(file_path)
            break       

def random_get_unuse_data_main():
    """
        从大数据集中 排除 已经使用过的数据, 
        从未使用的数据中随机筛选 n 个数据
        cp 到目标文件夹
    """
    big_dir = '/media/zsl/data/zsl_datasets/GF/gaofa_large_20w/text'
    already_use_dirs = ['/media/zsl/data/zsl_datasets/enocr_datasets/already_use_1_1_1',
                        '/media/zsl/data/zsl_datasets/enocr_datasets/already_use_1_3_3']

    save_dir = '/media/zsl/data/zsl_datasets/enocr_datasets/kernelGAN_hr_1_1_1_5785'

    select_n = 5785

    big_list = os.listdir(big_dir)
    already_use_list = []
    for dir in already_use_dirs:
        already_use_sub_list = os.listdir(dir)
        print('alreadu_use_sub_list:',len(already_use_sub_list))
        already_use_list.extend(already_use_sub_list)
    
    print('big_list:', len(big_list))
    print('already_use_list:',len(already_use_list))

    unuse_list = [i for i in big_list if i not in already_use_list]

    print('un_use_list:',len(already_use_list))

    random_select_list = random.sample(unuse_list, select_n)
    print('随机从未使用数据中筛选{}'.format(select_n))
    print(len(random_select_list))


    for unuse_file in random_select_list:
        unuse_file_path = os.path.join(big_dir, unuse_file)
        save_path = os.path.join(save_dir, unuse_file)
        shutil.copy(unuse_file_path, save_path)

def split_train_val_main():
    """
    将所有数据按比例分为训练集和验证集两部分
    """
    big_dir = "/media/zsl/data/zsl_datasets/enocr_datasets/temp"
    train_target = '/media/zsl/data/zsl_datasets/enocr_datasets/temp2/train'
    val_target = '/media/zsl/data/zsl_datasets/enocr_datasets/temp2/val'
    val_ratio = 0.2    # 验证集比例

    # 取每个子文件夹
    big_sub_list = os.listdir(big_dir)
    sub_dirs = []
    for sub in big_sub_list:
        sub_path = os.path.join(big_dir,sub)
        if os.path.isdir(sub_path):
            sub_dirs.append(sub_path)
    
    # 在train 和val 目录下建立子文件夹
    for sub_dir in sub_dirs:
        sub_dir_name = sub_dir.split('/')[-1]
        print(sub_dir_name)

        train_sub_dir = os.path.join(train_target, sub_dir_name)
        if not os.path.isdir(train_sub_dir): os.makedirs(train_sub_dir)

        val_sub_dir = os.path.join(val_target, sub_dir_name)
        if not os.path.isdir(val_sub_dir): os.makedirs(val_sub_dir)


    # 每个子文件大小相同, 文件名称相同, 从每个子文件夹中选取相同名称的文件.
    name_list = os.listdir(sub_dirs[0])
    val_num = int(len(name_list)*val_ratio)
    random_select_list = random.sample(name_list, val_num)
    print(len(random_select_list))

    # 先cp 选出来的val
    count_val = 0
    for select_file_name in random_select_list:
        select_file_basename = os.path.splitext(select_file_name)[0]
        # 遍历每个子文件, 按random出文件名, 移动sub_dir中的文件
        count_val += 1
        for sub_dir in sub_dirs:
            sub_list = os.listdir(sub_dir)   
            current_sub_dir_name = sub_dir.split('/')[-1]
            current_sub_ext = os.path.splitext(sub_list[0])[-1]   

            src_path = os.path.join(sub_dir, select_file_basename+current_sub_ext)
            target_val_path = os.path.join(val_target, current_sub_dir_name, select_file_basename+current_sub_ext)
            shutil.move(src_path, target_val_path)
            # shutil.copy(src_path, target_val_path)


    rest_list = [name for name in name_list if name not in random_select_list]
    count_train = 0
    for select_file_name in rest_list:
        select_file_basename = os.path.splitext(select_file_name)[0]
        count_train += 1
        for sub_dir in sub_dirs:
            sub_list = os.listdir(sub_dir)   
            current_sub_dir_name = sub_dir.split('/')[-1]
            current_sub_ext = os.path.splitext(sub_list[0])[-1]   
            src_path = os.path.join(sub_dir, select_file_basename+current_sub_ext)
            target_train_path = os.path.join(train_target, current_sub_dir_name, select_file_basename+current_sub_ext)
            shutil.move(src_path, target_train_path)
            # shutil.copy(src_path, target_train_path)

    print(count_train, count_val)


def get_unused_datas():
    big_dir = '/media/zsl/data/2021make_dataset/data_team_commit/整合Data_A数据集/Data_A数据集/text'
    used_dirs = ['/media/zsl/data/zsl_datasets/enocr_datasets/40495_1_3_3/train/text',
                '/media/zsl/data/zsl_datasets/enocr_datasets/40495_1_3_3/val/text',
                '/media/zsl/data/zsl_datasets/enocr_datasets/make_data_1104/data_a_text']

    unused_dir = '/media/zsl/data/zsl_datasets/enocr_datasets/make_data_1104/unsed_text'

    big_list = os.listdir(big_dir)

    used_list = []
    for used_dir in used_dirs:
        file_list = os.listdir(used_dir)
        print(len(file_list))
        used_list.extend(file_list)

    unused_list = [i for i in big_list if i not in used_list ]

    print('big:', len(big_list))
    print('used_list:',len(used_list))
    print('unused_list:',len(unused_list))

    count = 0
    for name in unused_list:
        imgfile = os.path.join(big_dir, name)
        if os.path.isfile(imgfile):
            target_file_path = os.path.join(unused_dir, name)
            shutil.copy(imgfile, target_file_path)
            count += 1
        else:
            print('big dir without file:', imgfile)
            continue
    print("total cp {} files.".format(count))

def seperate():
    ori_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/pcb_wacv_2019_lab/pcb_wacv_2019_ext_2"
    dis_img_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/pcb_wacv_2019_lab/pcb_wacv_2019_ext_2_sep/all_images"
    dis_label_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/pcb_wacv_2019_lab/pcb_wacv_2019_ext_2_sep/all_labels"

    for cur_dir, sub_dir, files in os.walk(ori_dir):
        if len(files)==0:
            continue
        else:
            for file in files:
                file_path = os.path.join(cur_dir,file)
                if file.endswith('.xml'):
                    dst_path = os.path.join(dis_label_dir, file)
                else:
                    dst_path = os.path.join(dis_img_dir, file)    
                shutil.copy(file_path, dst_path)

def test():
    ori_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/pcb_wacv_2019_lab/pcb_wacv_2019_ext_2_sep/all_images"
    dst_dir = "/media/zsl/data/zsl_datasets/PCB_dataset/pcb_wacv_2019_lab/pcb_wacv_2019_ext_2_sep/large_img"
    os.makedirs(dst_dir)
    # 取每个子文件夹
    ori_list = os.listdir(ori_dir)
    for ori in ori_list:
        ori_path = os.path.join(ori_dir,ori)
        img = cv2.imread(ori_path)
        print(img.shape)
        h,w,_ = img.shape
        if h > 1024 and w > 1024:
            dst_path = os.path.join(dst_dir,ori)
            shutil.copy(ori_path, dst_path)

def merge_dir():
    # sub_dir_list = ["/home/zsl/Music/1/1",
    #                 "/home/zsl/Music/1/2",
    #                 "/home/zsl/Music/1/3"]
    # merge_dir = "/home/zsl/Music/1/merge"
    sub_dir_list = ["/media/zsl/data/zsl_datasets/for_train/scheme1/717_base/crop2/aug_imgs",
                    "/media/zsl/data/zsl_datasets/for_train/scheme1/pcba_wav_base/crop2/aug_imgs"]
    merge_dir = "/media/zsl/data/zsl_datasets/for_train/scheme1/all_train2/aug_imgs"

    if not os.path.exists(merge_dir):
        os.makedirs(merge_dir, exist_ok=True)

    for sub_dir in sub_dir_list:
        sub_list = glob.glob(sub_dir+'/*')
        for path in sub_list:
            base_name = os.path.basename(path)
            dst_path = os.path.join(merge_dir, base_name)
            shutil.copy(path, dst_path)



def get_all_files():
    root_dir = '/media/zsl/data/zsl_datasets/717/basedata/defects_aligned_white_labeled'
    output_path = '/media/zsl/data/zsl_datasets/for_train/scheme1/717_base/all_717/labels'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for c_dir, sub_dir, files in os.walk(root_dir):

        if len(files) == 0:
            continue
        else:
            for file in files:
                src = os.path.join(c_dir, file)
                dst = os.path.join(output_path,file)
                shutil.copy(src, dst)

def mv_matchfile():
    """
        移除包含match_str中字符的文件.
    """
    match_strs = ["少件2022050901013_aligned",
                    "少件2022050901014_aligned",
                    "少件2022050901015_aligned",
                    "少件2022050901016_aligned",
                    "少件2022050901017_aligned",
                    "少件2022050901018_aligned",
                    "少件2022050901019_aligned"]
    dir = "/media/zsl/data/zsl_datasets/for_train/scheme1/all_train3/labels/val"
    dst_dir = "/media/zsl/data/zsl_datasets/for_train/scheme1/all_train3/labels/val_wrong"
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


def zooming(img, scale=0.1):
    img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    return img


def batch_resize():
    template_dir = "/media/zsl/data/workspace/codes/obj_detection/detection_pre/test/zsl_test_image/new/base2"
    resize_dir = "/media/zsl/data/workspace/codes/obj_detection/detection_pre/test/zsl_test_image/new/base2_resize0.1"

    if not os.path.exists(resize_dir):
        os.makedirs(resize_dir, exist_ok=True)

    t_list = glob.glob(template_dir+"/*")
    for t in t_list:
        img = cv2.imread(t)
        img = zooming(img, scale=0.1)
        t_p = os.path.join(resize_dir, os.path.basename(t))
        cv2.imwrite(t_p, img)


import datetime
# import fitz  # fitz就是pip install PyMuPDF
def pyMuPDF_fitz(pdfPath, imagePath):
    startTime_pdf2img = datetime.datetime.now()  # 开始时间

    print("PDFPath=" + pdfPath)
    print("imagePath=" + imagePath)

    pdfDoc = fitz.open(pdfPath)
    for pg in range(pdfDoc.pageCount):
        page = pdfDoc[pg]
        rotate = int(0)
        # 每个尺寸分辨率的缩放系数为zoom_x，zoom_y
        # 此处若是不做设置，默认图片大小为：792X612, dpi=96
        zoom_x = 3#1.33333333  # (1.33333333-->1056x816)   (2-->1584x1224)
        zoom_y = 3#1.33333333
        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pix = page.getPixmap(matrix=mat, alpha=False)

        if not os.path.exists(imagePath):  # 判断存放图片的文件夹是否存在
            os.makedirs(imagePath)  # 若图片文件夹不存在就创建
        name = pdfPath.split("/")[-1].split(".pdf")[0]
        pix.writePNG(imagePath + '/' + '%s_%s.png'%(name,pg))  # 将图片写入指定的文件夹内

    endTime_pdf2img = datetime.datetime.now()  # 结束时间
    print('pdf2img时间=', (endTime_pdf2img - startTime_pdf2img).seconds)


def pad2img_main():
    # 1、存放PDFs地址
    pdfPath = '/media/whwh/data/Model/X2img/x2img/pdf/*'
    # 2、需要储存图片的目录
    imagePath = '/media/whwh/data/Model/X2img/x2img/img/SR'

    pdfFiles = glob(pdfPath)
    for pdfFile in tqdm(pdfFiles):
        pyMuPDF_fitz(pdfFile, imagePath)


if __name__ == '__main__':
    # merge_dir()
    # rm_lr_nohr()
    # get_some4train()
    # sr_resize_save()
    # png_tran_jpg()
    # mv_same()
    # random_crop_img_main()
    # mv_diff()
    # mv_txt_same()
    # data_split()
    # assign_data_main()
    # check_split_merge()
    # data_split()
    # auto_split_change()   \
    rename_files()
    # check_file_exist()
    # random_get_unuse_data_main()
    # split_train_val_main()
    # get_unused_datas()
    # seperate()
    # test()
    # get_all_files()
    # mv_matchfile()
    # batch_resize()  #