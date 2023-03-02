from distutils import command
import shutil
import cv2
import os 
import glob

from pro_diff import zooming

"""
前一百张为样板图，后面每10张来一副缺陷图。
"""
def duplication():
    # template_p = "/media/zsl/data/workspace/codes/obj_detection/detection_pre/test/zsl_test_image/new/base/1.jpg"
    template_dir = "/media/zsl/data/workspace/codes/obj_detection/detection_pre/test/zsl_test_image/new/base2_resize0.1"
    target_dir = "/media/zsl/data/workspace/codes/obj_detection/detection_pre/test/zsl_test_image/new/frames2_0.1"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)    

    # 清空之前生成的
    try:
        commend = 'rm -r ' + target_dir + '/*'
        os.system(commend)
        print('use command:',commend)
        print('clear cache done')
    except:
        print('clear cache done')



    t_list = glob.glob(template_dir+"/*")
    print(t_list)

    times = [100,20,20, 20]
    k = 0
    time = 0
    for i, template_p in enumerate(t_list):
        time += times[i]
        while(k < time):
            if k < 10:
                file_name = '0000' + str(k)
            elif k < 100:
                file_name = '000' + str(k)
            elif k < 1000:
                file_name = '00' + str(k)
            elif k < 1000:
                file_name = '0' + str(k)
            else:
                file_name = str(k)
            target_p = os.path.join(target_dir, file_name+".jpg")
            shutil.copy(template_p, target_p)
            k += 1


def tran_video():
    cache_path = '/media/zsl/data/workspace/codes/obj_detection/detection_pre/test/zsl_test_image/new/frames2_0.1'
    video_path = '/media/zsl/data/workspace/codes/obj_detection/detection_pre/test/zsl_test_image/new/yoyoyo_t2_01.mp4'
    merge_commend = 'ffmpeg -i ' + cache_path + '/%05d.'+ 'jpg' + ' -c:v libx264 -vf \"fps=24,format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2\" ' + video_path
    os.system(merge_commend)

if __name__ == "__main__":
    duplication()   
    tran_video()