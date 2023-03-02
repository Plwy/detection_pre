from curses.ascii import ETB
from re import L
import cv2
import os
import numpy as np
import random
import pywt
import pywt.data
import glob
import matplotlib.pyplot as plt

def zooming(img, scale=0.1):
    img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    return img


def absdiff():
    """
        找茬游戏
    """
    img = cv2.imread("test/zsl_test_image/zhaocha/zhaocha2.png")
    h,w,c = img.shape
    print(h,w)
    half = int(w/2)
    img1 = img[:,2:half]
    img2 = img[:,half+1:w-2]
    cv2.imwrite('zhaocha_img1.png',img1)
    cv2.imwrite('zhaocha_img2.png',img2)

    print(img1.shape)
    print(img2.shape)

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    dst = cv2.absdiff(img1, img2)  ####
    cv2.imshow('dst', dst)
    cv2.waitKey(0)


def trace_boundary():
    """
    ref:https://blog.csdn.net/wxplol/article/details/73070791
    reimplement
    """
    pass


def cal_NCC(img1, img2):
    print(np.std(img2))
    Ncc = np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))
    return Ncc


def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))

def template_match_pipline(img1, img2):
    """
    Printed Circuit Board Assembly Defects Detection Using Image Processing Techniques2016
    1.convert rgb to gray
    2.Calculate NCC between template and reference image
    3.Split color image and do template matching on RCB
    4.Combine the result
    """

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    """
        1.模板图上滑动窗口, 每个窗口和测试图进行归一化互相关匹配.
        2.若最匹配的区域 与窗口交并比高,达1, 那么匹配成功.
        3.底层的归一化互相关进行匹配的代码.
    """
    x = np.zeros_like(img1)

    return x

def wave_tranform_pipline(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    img1_f = cv2.medianBlur(img1_gray, 1)
    img2_f = cv2.medianBlur(img2_gray, 1)
    # cv2.imshow("l",img1)
    # cv2.imshow("2",img2)
    # cv2.imshow("ll",img1_f)
    # cv2.imshow("22",img2_f)
    # cv2.waitKey(0)
    # # original = pywt.data.camera()
    # original = img1_f
    # print(original.shape)

    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
            'Vertical detail', 'Diagonal detail']
    # coeffs2 = pywt.dwt2(original, 'bior1.3')
    coeffs1 = pywt.dwt2(img1_f, 'bior1.3')
    coeffs2 = pywt.dwt2(img2_f, 'bior1.3')

    LL, (LH, HL, HH) = coeffs1
    LL2, (LH2, HL2, HH2) = coeffs2
    diff  = cv2.subtract(LL, LL2)
    # diff  = cv2.subtract(LH, LH2)
    # fig = plt.figure(figsize=(12, 3))
    # for i, a in enumerate([LL, LH, HL, HH]):
    #     ax = fig.add_subplot(1, 4, i + 1)
    #     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    #     ax.set_title(titles[i], fontsize=10)
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    # fig.tight_layout()
    # plt.show()
    return diff

def bg_sub_pipline1(imgt, imgd):
    """
    imgt: template img  
    imgd: defect img
    检测点对比了两副图的差异，测试效果取决于，图像大小，形态核，二值图阈值设定

    """
    kernel = np.ones((5,5),np.uint8)
    #Changing color space
    g_o_img = cv2.cvtColor(imgt, cv2.COLOR_BGR2LAB)   [...,0]
    g_def_img = cv2.cvtColor(imgd, cv2.COLOR_BGR2LAB)[...,0]
    #Image subtraction
    sub =cv2.subtract(g_o_img, g_def_img)
    # cv2.imshow('sub',sub)
    # cv2.waitKey(0)
    #Morphological opening 
    thresh = cv2.threshold(sub , 30, 255, cv2.THRESH_BINARY)[1]
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('open',opening)
    # cv2.waitKey(0)
    # xor
    im=cv2.bitwise_not(opening)

    # cv2.imshow('xor',im)
    # cv2.waitKey(0)
    #Detecting blobs
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByCircularity = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(im)

    #Drawing circle around blobs
    im_with_keypoints = cv2.drawKeypoints(imgd, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return im_with_keypoints


def bg_sub_pipline0(img):
    """
    Printed Circuit Board Assembly Defects Detection Using Image Processing Techniques2016
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    img_gray = cv2.cvtColor(img_hsv, cv2.COLOR_RGB2GRAY)
    img_f = cv2.medianBlur(img_gray, 3)
    ret, bin = cv2.threshold(img_f, 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('bin',bin)
    # cv2.waitKey(0)

    #close消除mask内的黑色噪声点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)

    return  bin


def canny_pipline(imgt, imgd):
    """ 
    Quality Control of PCB using ImageProcessing 2016

    canny 边缘检测后直接相减
    """
    imgt_gray = cv2.cvtColor(imgt, cv2.COLOR_RGB2GRAY)
    # imgt_gray = cv2.medianBlur(imgt_gray, 9)
    imgt_gray = cv2.GaussianBlur(imgt_gray, (5,5),cv2.BORDER_DEFAULT)
    # imgt_gray = cv2.blur(imgt_gray, 5)


    imgt_canny = cv2.Canny(imgt_gray, 50, 150, L2gradient=True)
    imgd_gray = cv2.cvtColor(imgd, cv2.COLOR_RGB2GRAY)
    # imgd_gray = cv2.medianBlur(imgd_gray, 9)
    imgd_gray = cv2.GaussianBlur(imgd_gray, (5,5),cv2.BORDER_DEFAULT)

    imgd_canny = cv2.Canny(imgd_gray, 50, 150, L2gradient=True)
    # diff = cv2.absdiff(imgt_canny, imgd_canny)
    diff = cv2.bitwise_xor(imgt_canny, imgd_canny)
    # cv2.imshow("diff", zooming(diff,0.2)  )
    # cv2.waitKey(0)  
    return diff

def contours_pipline(imgt, imgd):
    """

    """
    # t
    imgt_gray = cv2.cvtColor(imgt, cv2.COLOR_RGB2GRAY)
    imgt_gray = cv2.medianBlur(imgt_gray, 5)
    imgt_canny = cv2.Canny(imgt_gray, 50, 150, L2gradient=True)
    # contours_t, hierarchy_t = cv2.findContours(imgt_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # list : [(point_num, 1, 2)]

    blur = cv2.medianBlur(imgt_canny,5)
    th = cv2.threshold(blur, 254, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(th,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations = 2)
    contours_t, gihr = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#
    print("t contours len:",len(contours_t))

    # for c in contours_t:
    #     if cv2.contourArea(c) > 20:
    #         (x,y,w,h) = cv2.boundingRect(c)
    #         cv2.rectangle(imgt, (x,y),(x+w,y+h),(0,0,255),3)#图像加框，参数1：图像，参数2：左上角坐标，参数3：右下角坐标，参数4：框的颜色，参数5：框的粗细
    #         # cv2.rectangle(imgt,(250,250),(500,500),(255,255,0),2)
    #         if x > 100 and y > 100:
    #             cv2.rectangle(imgt,(x,y),(x+w,y+h),(238,44,44),8)

    cv2.drawContours(imgt, contours_t, -1, (255,0,0), 1)   


    imgd_gray = cv2.cvtColor(imgd, cv2.COLOR_RGB2GRAY)
    # imgd_gray = cv2.medianBlur(imgd_gray, 5)
    imgd_canny = cv2.Canny(imgd_gray, 50, 150, L2gradient=True)
    contours_d, hierarchy_d = cv2.findContours(imgd_canny, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # list : [(point_num, 1, 2)]
    print("d contours len:",len(contours_t))

    cv2.drawContours(imgd,contours_d, -1, (255,0,0), 1) 

    # diff_canny = cv2.absdiff(imgt_canny, imgd_canny)

    # 记录最匹配的值的大小和位置
    temp = imgd.copy()

    pair_pos = []
    min_value = 2
    for i in range(len(contours_t)):
        for j in range(len(contours_d)):
            value = cv2.matchShapes(contours_d[j],contours_t[i],1,0.0)
            # print(value)
            if value < min_value:
                min_value = value
                min_pos = j
        pair_pos.append((i, min_pos))
        # print(pair_pos, min_value)
        cv2.drawContours(temp,[contours_d[pair_pos[i][1]]],0,[255,0,0],3)

        # cv2.imshow("imgd", zooming(imgd))
        # cv2.waitKey(0)  
    return  temp


def HSL_pipline(imgt, imgd):
    """
        转HSL后减
    """
    imgt_hsl = cv2.cvtColor(imgt, cv2.COLOR_BGR2HLS)
    t_l = imgt_hsl[:,:,2]

    imgd_hsl = cv2.cvtColor(imgd, cv2.COLOR_BGR2HLS)
    d_l = imgd_hsl[:,:,2]

    # t_l_his = cv2.calcHist([t_l], [0], None, [256], [0, 256])
    
    _, t_l = cv2.threshold(t_l, 30, 255, cv2.THRESH_BINARY)
    _, d_l = cv2.threshold(d_l, 30, 255, cv2.THRESH_BINARY)

    sub_l = cv2.absdiff(t_l, d_l)

    return sub_l


def SSR(img, sigma=195):
    temp = cv2.GaussianBlur(img, (0,0), sigma)
    gau = np.where(temp==0, 0.01, temp)
    retinex = np.log10(img+0.01) - np.log10(gau)
    return retinex

def Retinex_pipline(imgt, imgd):
    #一幅给定的图像s(x,y)可以分解为两个不同的图像：反射图像R(x,y)和亮度图像L(x,y)。
    imgt_ssr1 = SSR(imgt,180)
    imgd_ssr2 = SSR(imgt,195)
    re = np.hstack((imgt_ssr1,imgd_ssr2))
    # cv2.imshow("imgt_ssr1",imgt_ssr1)
    # cv2.waitKey(0)
    # cv2.imwrite("imgt_ssr1.png",re)
    return re

def get_diff_main():
    """
        读入一对图；或者读入成对的文件夹
    """
    save_dir = "image_subtract"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)

    # imgt_p = "/media/zsl/data/workspace/codes/obj_detection/detection_pre/test/pair/多件样板.jpg"
    # # imgd_p = "/media/zsl/data/workspace/codes/obj_detection/detection_pre/test/pair/aglined_duojian.png"
    imgt_p = "test/zsl_test_image/template/1.jpg"
    imgd_p = "test/zsl_test_image/aligned/多件1_1_aligned.jpg"


    # # 去高光图
    # imgt_p = 'test/highlight_test/highlight0_result/xiufu_1.jpg'
    # imgd_p = 'test/highlight_test/highlight0_result/xiufu_多件1_0_aligned.jpg'

    # # 成对的元件切图
    # imgt_p = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/all_pairs/diff1/d/裸板-B001_aligned_0_88.jpg"
    # imgd_p = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/all_pairs/diff1/t/裸板-B001_88.jpg"
    imgt = cv2.imread(imgt_p)
    imgd = cv2.imread(imgd_p)
    # scale = 0.2 # 0.2
    # imgt = cv2.resize(imgt,(int(scale*imgt.shape[1]),int(scale*imgt.shape[0])))
    # imgd = cv2.resize(imgd,(int(scale*imgd.shape[1]),int(scale*imgd.shape[0])))
    
    sub_dir = os.path.join(save_dir, os.path.splitext(os.path.basename(imgd_p))[0])
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir,exist_ok=True)

    imgd_basename = os.path.basename(imgd_p)
    print(imgd_basename)
    name, ext = os.path.splitext(imgd_basename)
    print(name, ext)
    diff_dic = ['sub','HSVsub','blob','tempmatch','wave_tran', 'canny', 'contours', 'HSL']
    for i in range(8):
        print(i)
        diff = get_diff(imgt, imgd, mode=i)
        diff_basename = name + '_diff' + diff_dic[i] + ext
        save_path = os.path.join(sub_dir, diff_basename)
        cv2.imwrite(save_path, diff)


def get_diff(imgt, imgd, mode=0):
    """
        读入样板图和测试图，选择差分模式
    """
    if mode == 0:  # 直接减
        diff = cv2.absdiff(imgt,imgd)
    elif mode == 1: # 转HSV
        imgt_p = bg_sub_pipline0(imgt)
        imgd_p = bg_sub_pipline0(imgd)
        diff = cv2.absdiff(imgt_p, imgd_p)
    elif mode == 2:  # Blob分析,Display image with circle around defect
        diff = bg_sub_pipline1(imgt, imgd)
    elif mode == 3:     # 模板匹配
        ## template match
        diff = template_match_pipline(imgt,imgd)
    elif mode == 4:        # 小波变换
        diff = wave_tranform_pipline(imgt, imgd)
    elif mode == 5:     #canny边缘检测
        diff = canny_pipline(imgt,imgd)  
    elif mode == 6:       # 轮廓分析 
        diff = contours_pipline(imgt,imgd)
    elif mode == 7:    # 转HSL
        diff = HSL_pipline(imgt, imgd)
    elif mode == 8: # retinex 
        diff = Retinex_pipline(imgt, imgd)

    return diff


def get_diff_single(imgt_p, imgd_p, save_dir):
    """
        多种模式相减并保存
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)

    imgt = cv2.imread(imgt_p)
    imgd = cv2.imread(imgd_p)
    # scale = 0.2 # 0.2
    # imgt = cv2.resize(imgt,(int(scale*imgt.shape[1]),int(scale*imgt.shape[0])))
    # imgd = cv2.resize(imgd,(int(scale*imgd.shape[1]),int(scale*imgd.shape[0])))
    
    sub_dir = os.path.join(save_dir, os.path.splitext(os.path.basename(imgd_p))[0])
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir,exist_ok=True)

    imgd_basename = os.path.basename(imgd_p)
    print(imgd_basename)
    name, ext = os.path.splitext(imgd_basename)
    print(name, ext)
    diff_dic = ['sub','HSVsub','blob','tempmatch','wave_tran', 'canny', 'contours', 'HSL']
    for i in range(8):
        print(i)
        diff = get_diff(imgt, imgd, mode=i)
        diff_basename = name + '_diff' + diff_dic[i] + ext
        save_path = os.path.join(sub_dir, diff_basename)
        cv2.imwrite(save_path, diff)

def batch_get_diff():
    """
        多对图,多种模式,批量相减
    """
    imgt_dir = "/media/zsl/data/zsl_datasets/717/labeled/td_pairs/template"
    imgd_dir = "/media/zsl/data/zsl_datasets/717/labeled/td_pairs/aligned"
    save_dir = '/media/zsl/data/zsl_datasets/717/labeled/td_pairs/sub_result'
    imgd_paths = glob.glob(imgd_dir+'/*')

    for imgd_path in imgd_paths:
        t_num = os.path.basename(imgd_path).split('_')[0]  # 取模板编号
        imgt_name = 'template_'+t_num+'.jpg'
        imgt_path = os.path.join(imgt_dir, imgt_name)

        get_diff_single(imgt_path, imgd_path, save_dir)


def get_t_mask():
    """
        获取模板图pcba板区域的mask
    """
    imgt_p = "test/zsl_test_image/template/1.jpg"
    imgt = cv2.imread(imgt_p)
    imgt_hsl = cv2.cvtColor(imgt, cv2.COLOR_BGR2HLS)
    t_l = imgt_hsl[:,:,2]
    _, t_l = cv2.threshold(t_l,30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((37,37),np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55))  
    t_l = cv2.morphologyEx(t_l, cv2.MORPH_OPEN, kernel)
    # temp = t_l.copy()
    kernel = np.ones((37,37),np.uint8)

    t_l = cv2.morphologyEx(t_l, cv2.MORPH_CLOSE, kernel)
    contours , h = cv2.findContours(t_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max = 0
    max_c = []
    for c in contours:
        c_a = cv2.contourArea(c)
        if c_a > max:
            max = c_a
            max_c.append(c)
    print(max_c[-1].shape)
    b_c = max_c[-1]

    # b_c_ps = cv2.convexHull(b_c.astype(int))
    # cv2.fillPoly(mask, b_c_ps, (255,255,255),-1)
    mask = np.zeros_like(imgt)
    t_h,t_w,_ = imgt.shape
    for w in range(t_w):
        for h in range(t_h):
            result = cv2.pointPolygonTest(b_c, (w,h), False)
            if result >= 0:
                mask[h,w,:] = (255,255,255)
            else:
                mask[h,w,:] = (0,0,0)
    # mask = zooming(mask, 0.2)

    cv2.imwrite("1_mask.jpg",mask)


if __name__ == "__main__":
    # absdiff()
    # template_match()
    # get_diff_main()
    # get_t_mask()
    batch_get_diff()