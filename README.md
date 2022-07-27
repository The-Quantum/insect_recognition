# Insect pests recognition

This project is dodicated to the detection and the recognition of insects. 
The method consist of retraining [YOLOv5](https://github.com/ultralytics/yolov5) on [IP102](https://github.com/xpwu95/IP102) insects pests dataset. 

Insect pests are well known to be a major cause of damage to the commercially important agricultural crops [IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_IP102_A_Large-Scale_Benchmark_Dataset_for_Insect_Pest_Recognition_CVPR_2019_paper.pdf). Therefore, in the framework of YOLOv5 test on the substentially different dataset [IP102] represent a very good choice. 

[IP102] dataset contains more than 75.000 images belonging to 102 categories, which exhibit a natural long-tailed distribution. In
addition, we annotate about 19.000 images with bounding boxes for object detection. 

# Data preparation

## YOLOv5 data format
The YOLOv5 annotation format for a given object presents in an image is the following :
```
 class center_X center_y width height
```
Obviously, ```class``` refers to the object's class number while the rest of the annotation refer to center coordinate of the object's bounding box in the image and the width and height of the corresponding bounding box. 
For each image, the contained object's bounding box annotation is saved as a single line in a text file with the same file name as the image containt the object ***of course without image extension but .txt instead***. In the case there are multiple objects in the same image, there is one annotation text file having as many line as there objects in the image each of which is realted to one object.
<p align="center"><img width="800" alt="PR_step1" src="https://user-images.githubusercontent.com/26833433/122260847-08be2600-ced4-11eb-828b-8287ace4136c.png"></p>

## IP102 data format
 
so you could write a script on your own that does that for you.