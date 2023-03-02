## Inroduction
检测相关代码及可执行函数说明.


### 传统方法相关
**图像对齐**

```bash
python image_align.py  # 
```
- keep_template_kp() 检测并保存特征点
- batch_align_main2()批量特征点检测并对齐

**图像差分**

```bash
python image_subtract.py #
python pro_diff.py  # 多种差分模式,批量相减并保存
```

**生成序列图像并转为视频**

```bash
python make_video.py  # 
```
生成用于mog建模的视频

**去除高亮**

```bash
python mvhighlight.py  
```
针对高亮焊点,用图像修复的方法去除高亮
               
                                                          

### 目标检测相关

#### 当前使用流程

```
python compare_flow.py
```

1.flow_main()    对待测图批量进行切图

2.将切图文件输入到检测模型进行检测,得到labels

3.对切图的labels做拼接. 

- batch_stitch_template()   拼模板图
- batch_stitch_defect()  拼测试图

4.对拼接后的检测结果做对比
```bash
python compare.py 
```
测试图和模板图的结果做对比, 生成拼接并做标注的检测结果.
```
python get_validate.py   
```
检测结果和gt标注做对比,生成P,R,F1 , 并保存检错图.





#### 可用工具

**获得数据的分类类别切图**

```
python get_all_cals.py
```
解析后得到类别分布, 或按类别保留目标小图.


**随机贴图/贴大图**
```bash
# patch_main()设置一张图的最大贴图数， 不考虑重叠的贴图; 随机贴图random_patch_img_test()
python patch_pro.py  
# 先随机贴大图1, 再随机贴大图2, 再随机贴小图,每次都直到贴不下.
python patch_img.py 
# patch_issue_main()在切图贴图后，添加原图上未覆盖的标签
python patch_issue.py  
```

**滑窗切图**
```bash
python image_crop.py  # 
```
训练集制作使用,切图同时切标签.

```bash
python stitch_test.py 
```

crop_main()  测试集切图,不需要标签


**小图检测结果 拼接成大图**
```bash
python stitch_test.py # 
```
- stitch_anno_main()  标签拼接;  
- stitch_main() 拼图像


**检测结果对比可视化**
```bash
python compare.py  #

```
- batch_compare() 批量对比测试图检测结果和模板图检测结果,并可视化差异.
- compare_test() 单张测试调试函数时使用

**大目标为中心切图**
```bash
python big_obj.py   # 
```
- main() 整合了整个流程:抠取mask,切图,resize后选取大目标, 然后以大目标为中心随机切图.


**画标注**
```bash
python draw_labels.py #    
```
- draw_crop_line_test()画出切图时候 的切线.

**文本解析和转换**
```bash
python parse_objfile.py # 文本解析
```
- json_parse() json文本解析
- txt_parse() txt文本解析
- xml_parse() xml文本解析

all return:  obj_list [[x,y,x,y, classname, conf],...]

```bash
python convert.py
```
to_yolo_main()   输入 xml or txt 文件 , xyxy tran to xywh yolo
json2xml_main() 
txt2xml_main() 

