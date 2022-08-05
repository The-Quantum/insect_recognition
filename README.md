# Train YOLOv5 for `Insect pests` recognition

This project is dodicated to the detection and the recognition of insects. 
The method consist of retraining [YOLOv5](https://github.com/ultralytics/yolov5) on [IP102](https://github.com/xpwu95/IP102) insects pests dataset. 

Insect pests are well known to be a major cause of damage to the commercially important agricultural crops [IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_IP102_A_Large-Scale_Benchmark_Dataset_for_Insect_Pest_Recognition_CVPR_2019_paper.pdf). Therefore, in the framework of YOLOv5 test on the substentially different dataset [IP102] represent a very good choice. 

# Set virtual environnement [TO DO]

# Data preparation

## YOLOv5 data format
The YOLOv5 annotation format for a given object presents in an image is the following :
```
 class center_X center_y width height
```
Obviously, ```class``` refers to the object's class number while the rest of the annotation refer to center coordinate of the object's bounding box in the image and the width and height of the corresponding bounding box. 

For each image, the contained object's bounding box annotation is saved as a single line in a text file with the same file name as the image containt the object ***of course without image extension but .txt instead***. In the case there are multiple objects in the same image, there is one annotation text file having as many line as there objects in the image each of which is realted to one object.
<div align="center">YOLOv5 annotation illustration</div>
<p align="center"><img width="800" alt="PR_step1" src="https://github.com/The-Quantum/insect_recognition/blob/main/notebook/Yolo_annotations_illustration.png"></p>

In the example of the previous figure, the annototion file will containt **4 lines** corresponding to the four objects. 

To train YOLOv5 on a given dataset, the prior requirement is to prepare the annotations correspondingly to the above described Yolo format. 

## IP102 data format
[IP102] dataset contains more than 75.000 images belonging to 102 categories, which exhibit a natural long-tailed distribution. About 19.000 images are annotated with bounding boxes for object detection. However, the annotation are save into `xml` files which are note compatible with YOLO. Each `xml` file prevides information of the bounding boxes containing the insect in the image as well as the insect class and the filename and size `(width, hight, depth)` of the corresponding image. In case a given image contains many insects, the corresponding `xml` annotation file provide as much bounding box as there are insects.

The bounding box coordinate are provide in the format ```x_min, y_min, x_main, y_max``` where `(x_min, y_mim)` are the coordinates of the to left corner of the bounding boxe and `(x_min, y_mim)` that of the bottom right corner. 

Therefore, it is require to write a module that adapt this annotation into YOLOv5 format. This is the purpose of `prepare_annotation.py` module.

To run the annotations module, first download [IP102 v1.1](https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing) the data set and the corresponding annotations. When you follow the link, you have to choose between classification and dectection dataset. Make sure to choose detection data as it is the purpose of this repository.
Then unzip both `JPEGImages.tar` and `Annotations.tar` in the `datasets/`. To do so the simplest way is to copy both file into `datasets/`, then navigates into it and apply the following untar code.
```
 tar -xvf Annotations.tar & tar -xvf JPEGImages.tar
```

Then step into the created `Annotations/` dir and run the following code.
```
 rm IP087000986.xml
```
The reason we remove this file is that its content organisation is differement from all other file. I decided no to spend time to adapt `prepare_annotation.py` to a unique file. Moreover its content present two bboxes while the image it is related to contains only one insect.

Now, run the following code from the `root/` 
```
 python prepare_annotation.py
```
You will see the timeline of the reformatation process. This code will create `labels` folder and located the formated annotations. 

The last think to do concerning the data set should have been to download `classes.txt`[https://github.com/xpwu95/IP102/blob/master/classes.txt] file which containts the 102 classes of insects. I have done that and you can find this file in `datasets/`.

## TRAIN TEST VAL SPLIT
To train  YOLOv5 the data set have to be split into `train/`, `val/` and `test/` folder. Each of this file folder should contains two subfolders including `images/` and `labels/`. The modules `python train_test_val_split.py` does this properly.
```
 python train_test_val_split.py
```
This will first create all the require folders and subfolders within `dataset/` parent folder. Then, shuffles and splits the images form `JPECImages/` into `train/`, `val/`, `test/` subset in proportion of `0.6, 0.2, 0.2`. Then copy each image or labels of each set to the corresponding `images/` or `labels/` folder.

# YOLOv5 training of IP102 dataset
