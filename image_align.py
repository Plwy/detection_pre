import cv2
import os
import numpy as np
import time
import glob
import joblib
from tqdm import tqdm

def alignImages(im1, im2):
    '''
    - 特征检测,特征描述子
    - 特征匹配
    - 单应性矩阵计算，图像映射
    '''
    MAX_MATCHES = 500
    # GOOD_MATCH_PERCENT = 0.15
    GOOD_MATCH_PERCENT = 0.1

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    # orb = cv2.ORB_create(MAX_MATCHES)
    # keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
    # keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)

    # sift = cv2.xfeatures2d.SIFT_create()
    # keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
    # keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)

    akaze = cv2.AKAZE_create()
    keypoints1, descriptors1 = akaze.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = akaze.detectAndCompute(im2Gray, None)
    
    # Match features.汉明顿距离比较匹配点的相匹配度
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score. 
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    
    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imshow('match', imMatches)    
    cv2.waitKey(0)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    
    return im1Reg, h


def align_main():
    # img = cv2.imread('zsl_test_image/多件1.jpg')
    st = time.time()
    img = cv2.imread('/media/zsl/data/temp/裸板-B001.jpg')
    back = cv2.imread('/media/zsl/data/temp/裸板-B002.jpg')
    save_path = '/media/zsl/data/temp/裸板-B002.jpg'

    print(img.shape)
    print(back.shape)
    scale = 0.05
    # img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    # back = cv2.resize(back,(int(scale*back.shape[1]),int(scale*back.shape[0])))
    alignedimg, h = alignImages(img, back)
    cv2.imwrite(save_path, alignedimg)
    print(time.time()-st)



# keyPoint以字典形式存储转为KeyPoint
def _pickle_keypoint(keypoint):  # : cv2.KeyPoint
    return cv2.KeyPoint(
        keypoint['pt'][0],
        keypoint['pt'][1],
        keypoint['size'],
        keypoint['angle'],
        keypoint['response'],
        keypoint['octave'],
        keypoint['class_id'],
    )


 # cv2.KeyPoint转为普通点格式
def _dump(keypoints):

   return [
       {"pt": p.pt,
        "size": p.size,
        "angle": p.angle,
        "response": p.response,
        "octave": p.octave,
        "class_id": p.class_id
       } for p in keypoints
   ]

def make_key_dump(kp, des, kp_path, compress=3):
    """
        保存特征点为.pkl格式
    """
    kp_dump = _dump(kp)
    dump_data = (kp_dump, des)
    joblib.dump(dump_data, kp_path, compress=compress)



def load_temp_kps(kp_path):
    """
        返回kps 格式为cv2.KeyPoint
    """
    datas = joblib.load(kp_path)
    kps = [_pickle_keypoint(ekp) for ekp in datas[0]]
    des = datas[1]

    return kps, des


def keep_template_kp():
    """
        批量输入模板，检测特征点并保存
    """
    # 样板目录
    t_img_dir = "/media/zsl/data/zsl_datasets/717/basedata/1/t" 
    kp_save_dir = "/media/zsl/data/zsl_datasets/717/basedata/1/t_kps" 
    t_paths = glob.glob(t_img_dir+"/*")

    akaze = cv2.AKAZE_create()

    for t_path in tqdm(t_paths):
        t_name = os.path.splitext(os.path.basename(t_path))[0]
        
        t_img = cv2.imread(t_path)
        kp_path = os.path.join(kp_save_dir, t_name+'_kp.pkl')
        
        kps, des = akaze.detectAndCompute(t_img, None)
        # save kps
        make_key_dump(kps, des, kp_path)
    
    print("Template kps save under :",kp_save_dir)

def get_kps(img, mode=0):
    if mode == 0:
        akaze = cv2.AKAZE_create()
        kps, des = akaze.detectAndCompute(img, None)
    elif mode == 1:
        orb = cv2.ORB_create()
        kps, des = orb.detectAndCompute(img, None)

    else:
        print("not surport such mode,try 0 or 1!")
    return kps, des


def batch_align_main():
    """遍历检测板，与样板对齐
        如果样板图有kp文件则直接读取，否则检测特征点
    """
    t_img_dir = "/media/zsl/data/zsl_datasets/717/labeled/td_pairs/template"
    d_img_dir = "/media/zsl/data/zsl_datasets/717/labeled/td_pairs/defect"
    aligned_dir = "/media/zsl/data/zsl_datasets/717/labeled/td_pairs/aligned"

    GOOD_MATCH_PERCENT = 0.1

    # 需要根据文件名，在样板文件夹找对应的样板图
    d_paths = glob.glob(d_img_dir+'/*')
    
    for d_img_path in tqdm(d_paths):
        d_basename = os.path.basename(d_img_path)
        d_name,d_ext = os.path.splitext(d_basename)
        d_img = cv2.imread(d_img_path)

        ## 根据测试图的名字,找对应的模板图
        # t_name = d_name.split('_')[0][-1]
        # t_kp_path = os.path.join(t_img_dir, t_name+'_kp.pkl')
        # print("Template kp file path：", t_kp_path)
        # t_img_path = os.path.join(t_img_dir,t_name+'.jpg')
        # t_img = cv2.imread(t_img_path)
        t_name = 'template_' + d_name.split('_')[0]
        t_kp_path = os.path.join(t_img_dir, t_name+'_kp.pkl')
        print("Template kp file path：", t_kp_path)
        t_img_path = os.path.join(t_img_dir,t_name+'.jpg')
        t_img = cv2.imread(t_img_path) 

        # 获取样板特征点
        if not os.path.isfile(t_kp_path):
            print("Can not find template kp file!")
            print("Detect kp from img now!")
            if not os.path.isfile(t_img_path):
                print("Can not find template img!")
                print("Please check d_img filename is: template_name+\"_num\"+\".jpg\"")
            else:
                t_kps, t_des = get_kps(t_img, mode=0)

        else:
            t_kps, t_des = load_temp_kps(t_kp_path)

        
        # 获取测试板特征点
        d_kps, d_des = get_kps(d_img, mode=0)


        # 特征匹配和透视变化
        # Match features.汉明顿距离比较匹配点的相匹配度
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(t_des, d_des, None)
    
        matches.sort(key=lambda x: x.distance, reverse=False)
        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        
        # # Draw top matches
        # imMatches = cv2.drawMatches(t_img, t_kps, d_img, d_kps, matches, None)
        # cv2.imshow('match', imMatches)    
        # cv2.waitKey(0)
        
        # Extract location of good matches
        points_t = np.zeros((len(matches), 2), dtype=np.float32)
        points_d = np.zeros((len(matches), 2), dtype=np.float32)
        
        for i, match in enumerate(matches):
            points_t[i, :] = t_kps[match.queryIdx].pt
            points_d[i, :] = d_kps[match.trainIdx].pt
        
        # Find homography
        h, mask = cv2.findHomography(points_d, points_t, cv2.RANSAC)
        # Use homography
        height, width, channels = t_img.shape
        aligned_img = cv2.warpPerspective(d_img, h, (width, height))
        
        aligned_save_path = os.path.join(aligned_dir, d_name+'_aligned'+ d_ext)
        print(aligned_save_path)
        cv2.imwrite(aligned_save_path, aligned_img)



def batch_align_main2():
    """
    遍历检测板，与样板对齐
    直接读取kp文件
    templates_kps: 1_kp.pkl,2_kp.pkl,..
    detectes: 1/detect1.jpg,...; 2/detect2.jpg,...
    """
    t_img_dir = "/media/zsl/data/zsl_datasets/717/basedata/1/t_kps"
    d_img_dir = "/media/zsl/data/zsl_datasets/717/basedata/1/imgs"
    aligned_dir = "/media/zsl/data/zsl_datasets/717/basedata/1/dd"

    GOOD_MATCH_PERCENT = 0.1

    # 需要根据文件名，在样板文件夹找对应的样板图
    # d_paths = glob.glob(d_img_dir+'/*')
    for cur_dir, sub_dir, d_files in os.walk(d_img_dir):
        print(cur_dir, sub_dir, d_files)
        if len(d_files)!= 0:
            t_name = cur_dir.split('/')[-1]
            d_sub_dir = os.path.join(aligned_dir, t_name)
            if not os.path.exists(d_sub_dir):
                os.makedirs(d_sub_dir, exist_ok=True)

            t_kp_path = os.path.join(t_img_dir, t_name+'_kp.pkl')
            # 获取样板特征点
            t_kps, t_des = load_temp_kps(t_kp_path)
        else:
            continue

        for d_file in tqdm(d_files):
            d_name, d_ext = os.path.splitext(d_file)
            d_img_path = os.path.join(cur_dir, d_file)
            d_img = cv2.imread(d_img_path)

            print("detect fileis:", d_name)
            print("Template kp file path：", t_kp_path)

            # 获取测试板特征点
            d_kps, d_des = get_kps(d_img, mode=0)
            # 特征匹配和透视变化
            # Match features.汉明顿距离比较匹配点的相匹配度
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(t_des, d_des, None)
        
            matches.sort(key=lambda x: x.distance, reverse=False)
            # Remove not so good matches
            numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
            matches = matches[:numGoodMatches]

            # Extract location of good matches
            points_t = np.zeros((len(matches), 2), dtype=np.float32)
            points_d = np.zeros((len(matches), 2), dtype=np.float32)
            
            for i, match in enumerate(matches):
                points_t[i, :] = t_kps[match.queryIdx].pt
                points_d[i, :] = d_kps[match.trainIdx].pt
            
            # Find homography
            h, mask = cv2.findHomography(points_d, points_t, cv2.RANSAC)

            # Use homography
            height, width, channels = d_img.shape
            aligned_img = cv2.warpPerspective(d_img, h, (width, height),borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(255,255,255))
            
            aligned_save_path = os.path.join(d_sub_dir, d_name+'_aligned'+ d_ext)
            print(aligned_save_path)
            cv2.imwrite(aligned_save_path, aligned_img)

    print("yoyoyo3")


if __name__ == '__main__':
    # align_main()   
    keep_template_kp()  # 样板特征点检测并保存
    # batch_align_main()
    batch_align_main2()