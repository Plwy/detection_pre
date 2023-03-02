import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import os
import glob
from tqdm import tqdm


from PIL import Image
from numpy import average, dot, linalg

def subtract2(fg, bg):
    """
        转灰度后减
    """
    fg_gray=cv2.cvtColor(fg,cv2.COLOR_BGR2GRAY)
    bg_gray=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)

    dif=np.absolute(np.matrix(np.int16(bg_gray))-np.matrix(np.int16(fg_gray)))
    dif[dif>255]=255
    Difference=np.uint8(dif)

    cv2.imshow('Difference',Difference)
    cv2.waitKey(0)
    # cv2.imwrite('zsl_test_results/sub_aglined_quejian_TEST01.png',Difference)


def subtract3(fg, bg):
    """
        转灰度后处理了减
    """
    Grayscaled_Difference_path = "test/sutract_result/Grayscaled_Difference2.png"
    ori_Difference_path = "test/sutract_result/ori_Difference2.png"
    Threshold_Difference_path = "test/sutract_result/Threshold_Applied_Grayscaled_Difference2.png"
    Gaussian_Blur_path = "test/sutract_result/Gaussian_Blur2.png"

    fg_gray=cv2.cvtColor(fg,cv2.COLOR_BGR2GRAY)
    fg_gauss=cv2.GaussianBlur(fg_gray, (5, 5), 0)
    bg_gray=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
    bg_gauss = cv2.GaussianBlur(bg_gray, (5, 5), 0)

    #Difference between first frame and the rest in grayscale
    gray_diff=cv2.absdiff(fg_gray,bg_gray)
    #Difference between first frame and the rest in color
    difference=cv2.absdiff(fg,bg)
    #setting threshold for how clear we want the image to be (Values 0(black)-255(white))
    _, thresh_gray_diff=cv2.threshold(gray_diff,25,255,cv2.THRESH_BINARY)
    #difference for gaussian blur
    gaus_diff=cv2.absdiff(fg_gauss,bg_gauss)
    #displaying
    cv2.imwrite(ori_Difference_path, difference)
    cv2.imwrite(Grayscaled_Difference_path , gray_diff)
    cv2.imwrite(Threshold_Difference_path, thresh_gray_diff)
    cv2.imwrite(Gaussian_Blur_path, gaus_diff)


def tran_single_channel(img):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    return b, g, r

def sub_single(fg, bg):
    dif=np.absolute(np.matrix(np.int16(fg))-np.matrix(np.int16(bg)))
    dif[dif>255]=255
    Difference=np.uint8(dif)
    return Difference

def rgb_subtract4(fg, bg):
    """
        rgb单通道减
    """
    fg_b, fg_g, fg_r = tran_single_channel(fg)   # 得到rgb通道图
    bg_b, bg_g, bg_r = tran_single_channel(bg)

    diff_b = cv2.absdiff(fg_b,bg_b)      # 与样板通道图减 ,两种减法结果一致
    # diff_b = sub_single(fg_b,bg_b)     
    diff_g = cv2.absdiff(fg_g,bg_g)  
    diff_r = cv2.absdiff(fg_r,bg_r)  

    # print(fg_b.shape)
    # cv2.imshow('fg_b',fg_b)
    # cv2.imshow('bg_b',bg_b)
    # # cv2.imshow('g',fg_g)
    # # cv2.imshow('r',fg_r)
    # cv2.imshow('diff_b',diff_b)
    # cv2.waitKey(0)

    x = cv2.absdiff(diff_b, diff_g)
    cv2.imwrite('diff_b_g.png', x)   # rgb相减结果是否一致？不一致
    cv2.imwrite('aglined_quejian_TEST01_diff_b.png', diff_b)
    cv2.imwrite('aglined_quejian_TEST01_diff_g.png', diff_g)
    cv2.imwrite('aglined_quejian_TEST01_diff_r.png', diff_r)

# def bg_sub_pre(img):
#     img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     img_gray = cv2.cvtColor(img_hsv,cv2.COLOR_BGR2GRAY)
#     img_filted = cv2.medianBlur(img_gray,3, img_gray)


def hist_pro(img):
    """
        直方图是图像中像素强度分布的图形表达方式。
        它统计了每一个强度值所具有的像素个数。
        不同的图像的直方图可能是相同的
    """
    # # # 全局直方图均衡
    # hist_balance= cv2.equalizeHist(img)   

    # # 局部直方图均衡
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    hist_balance = clahe.apply(img)

    return hist_balance

def base_hist_subtract(fg, bg, mask):
    fg_gray=cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    bg_gray=cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    fg_balance = hist_pro(fg_gray)
    bg_balance = hist_pro(bg_gray)
    
    fg_ori_hist = cv2.calcHist([fg_gray], [0], mask, [256], [0, 256])
    bg_ori_hist = cv2.calcHist([bg_gray], [0], mask, [256], [0, 256])
    fg_balance_hist = cv2.calcHist([fg_balance], [0], mask, [256], [0, 256])
    bg_balance_hist = cv2.calcHist([bg_balance], [0], mask, [256], [0, 256])

    # hist_compare_fofb = cv2.compareHist(fg_ori_hist, fg_balance_hist, cv2.HISTCMP_CORREL) # 应该相似度大
    # hist_compare_fobo = cv2.compareHist(fg_ori_hist, bg_ori_hist, cv2.HISTCMP_CORREL)       # 
    # hist_compare_fobb = cv2.compareHist(fg_ori_hist, bg_balance_hist, cv2.HISTCMP_CORREL)  # 得到准确的差异 
    # hist_compare_bobb = cv2.compareHist(bg_ori_hist, bg_balance_hist, cv2.HISTCMP_CORREL)   # 应该相似度大
    # hist_compare_fbbb = cv2.compareHist(fg_balance_hist, bg_balance_hist, cv2.HISTCMP_CORREL)   # 应该相似度大
    # hist_compare_bofb = cv2.compareHist(bg_ori_hist, fg_balance_hist, cv2.HISTCMP_CORREL)   # 应该相似度大

    # print("his compare fg_ori_hist, fg_balance_hist:",hist_compare_fofb)
    # print("his compare fg_ori_hist, bg_ori_hist:",hist_compare_fobo)
    # print("his compare fg_ori_hist, bg_balance_hist:",hist_compare_fobb)
    # print("his compare bg_ori_hist, bg_balance_hist:",hist_compare_bobb)
    # print("his compare fg_balance_hist, bg_balance_hist:",hist_compare_fbbb)
    # print("his compare bg_ori_hist, fg_balance_hist:",hist_compare_bofb)
	# # CV_COMP_CORREL
	# # CV_COMP_CHISQR
	# # CV_COMP_INTERSECT
	# # CV_COMP_BHATTACHARYYA
	
    # # 绘制处理前后直方图
    plt.figure(figsize=(10,8),dpi=100)
    # plt.subplot(221),plt.imshow(fg[:,:,::-1]),plt.title('ori_fg')
    plt.subplot(221),plt.imshow(fg_gray,'gray'),plt.title('ori_fg')
    # plt.subplot(222), plt.hist(fg_ori_hist.ravel(), 256, [0, 256],color='r'),plt.title('ori_fg')
    plt.subplot(222), plt.plot(fg_ori_hist),plt.title('ori_fg')
    plt.subplot(223),plt.imshow(fg_balance,'gray'),plt.title('hist_balance_fg')
    plt.subplot(224),plt.plot(fg_balance_hist),plt.title('hist_balance_fg')
    plt.grid()
    # plt.show()


    plt.figure(figsize=(10,8),dpi=100)
    # plt.subplot(221),plt.imshow(bg[:,:,::-1]),plt.title('ori_fg')
    plt.subplot(221),plt.imshow(bg_gray,'gray'),plt.title('ori_bg')
    plt.subplot(222),plt.plot(bg_ori_hist),plt.title('hist_ori_bg')
    plt.subplot(223),plt.imshow(bg_balance,'gray'),plt.title('hist_balance_bg')
    plt.subplot(224),plt.plot(bg_balance_hist),plt.title('hist_balance_bg')
    plt.grid()
    plt.show()

    sub = cv2.absdiff(fg_balance,bg_balance)

    # cv2.imshow('banlance_sub',sub)
    # cv2.waitKey(0)
    cv2.imwrite('his_banlance_sub.png', sub)


def get_hist_compare(fg, bg):
    fg_gray=cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    bg_gray=cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    fg_ori_hist = cv2.calcHist([fg_gray], [0], None, [256], [0, 256])
    bg_ori_hist = cv2.calcHist([bg_gray], [0], None, [256], [0, 256])

    result = cv2.compareHist(fg_ori_hist, bg_ori_hist, cv2.HISTCMP_CORREL) # 应该相似度大
    
    return result


def mse(imageA, imageB):
	# 计算两张图片的MSE指标
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# 返回结果，该值越小越好
	return err


def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree



def get_thum(image, size=(64, 64), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image

# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    # opencv tran PIL
    image1 = Image.fromarray(cv2.cvtColor(image1,cv2.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv2.cvtColor(image2,cv2.COLOR_BGR2RGB))

    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res


def batch_pair_test():
    """
        处理来自于template 和 测试图的成对图像,根据指标比较相似度
    """
    t_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/all_pairs/t"
    d_dir = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/all_pairs/d"

    diff_dir = '/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/all_pairs/diff2'

    d_list = glob.glob(d_dir+'/*')

    com_result = {}
    for d_path in tqdm(d_list):
        d_basename = os.path.basename(d_path)
        d_name = os.path.splitext(d_basename)[0]
        # print(t_path)
        temp = d_basename.split('_')
        t_basename = temp[0]+'_'+temp[-1]
        t_path = os.path.join(t_dir, t_basename)

        # print(d_path)
        t_img = cv2.imread(t_path)
        d_img = cv2.imread(d_path)

        # cv2.imshow('0',t_img)
        # cv2.imshow('1',d_img)
        # cv2.waitKey(0)

        # 处理成对的图片
        # result = get_hist_compare(d_img, t_img)   # 直方图相关性
        result = calculate(d_img, t_img)          # 直方图形状

        result = image_similarity_vectors_via_numpy(d_img,t_img)  # 余弦相似度


        # if result > 0.5:
        #     print("OK:",result)
        #     continue
        # else:
        #     com_result[d_basename] = result
        com_result[d_basename] = result

    print(com_result)
    t = sorted(com_result.items(), key=lambda x:(x[1]))
    # print(t[:3])

    c = 0
    thd = 10
    for a in t:
        c+=1
        if c > thd:
            break
        result = com_result[a[0]]
        print(a[0],":", result)

        diff_d_name = a[0]
        temp = diff_d_name.split('_')
        diff_t_name = temp[0]+'_'+temp[-1]
        
        diff_t_img = cv2.imread(os.path.join(t_dir,diff_t_name))
        diff_d_img = cv2.imread(os.path.join(d_dir, diff_d_name))

        cv2.imwrite(os.path.join(diff_dir, 't',diff_t_name), diff_t_img )
        cv2.imwrite(os.path.join(diff_dir, 'd',diff_d_name), diff_d_img)



    # # print("NG:",result)
    # cv2.imwrite(os.path.join(diff_dir, diff_t_name), d_img)
    # cv2.imwrite(os.path.join(diff_dir, diff_d_name), t_img)

def main():
    """
        比较模板图和测试图的hist分布
    """
    imgt_p = "test/zsl_test_image/template/1.jpg"
    imgt = cv2.imread(imgt_p)
    imgd_p = "test/zsl_test_image/aligned/多件1_1_aligned.jpg"
    imgd = cv2.imread(imgd_p)

    mask_p = "test/zsl_test_image/template/1_mask.jpg"
    mask =  cv2.imread(mask_p)
    print(mask.shape, imgd.shape, imgt.shape)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    base_hist_subtract(imgd, imgt, mask)




if __name__ == "__main__":
    # # Path of Images
    # fg_path = "test/zsl_test_results/aglined_duojian.png"
    # # fg_path = "/home/zsl/Pictures/zhaocha2.png"

    # # bg_path = "zsl_test_results/aglined_duojian.png"
    # bg_path = "zsl_test_image/多件样板.jpg"
    # fg_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/pairs/IC_717/d/IC_717_d_5.jpg"
    # bg_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/pairs/IC_717/t/IC_717_t_5.jpg"
    # fg_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/pairs/IC_717/d/IC_717_d_9.jpg"
    # bg_path = "/media/zsl/data/workspace/codes/obj_detection/datasets/pcba_comp_dataset_plus/croped_lab/Ano_merge/pairs/IC_717/t/IC_717_t_9.jpg"

    # # h,w,c = img_defect.shape


    # fg = cv2.imread(fg_path)
    # bg = cv2.imread(bg_path)
    # # scale = 1
    # # if scale != 1:
    # #     fg = cv2.resize(fg,(int(scale*fg.shape[1]),int(scale*fg.shape[0])))
    # #     bg = cv2.resize(bg,(int(scale*bg.shape[1]),int(scale*bg.shape[0])))
    
    # # subtract2(fg, bg)
    # # subtract3(fg, bg)
    # # rgb_subtract4(fg, bg)
    # # bg_subtract_2016(fg,bg)

    # batch_pair_test()

    main()