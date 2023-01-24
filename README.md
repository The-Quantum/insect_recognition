# Train YOLOv5 for `Insect pests` recognition

This project is dedicated to the detection and the recognition of insects using YOLOv5. 
The method consist of retraining [YOLOv5](https://github.com/ultralytics/yolov5) on [IP102](https://github.com/xpwu95/IP102) insects pests dataset. 

Insect pests are well known to be a major cause of damage to the agricultural crops with important commercial losses[IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_IP102_A_Large-Scale_Benchmark_Dataset_for_Insect_Pest_Recognition_CVPR_2019_paper.pdf). Our goal here is to retrained `YOLOv5` that has proven state or the art efficiency on object detection dataset on a completely different type of dataset it has originaly been train on such as [IP102]. We will then evaluate the performances of the new retrained YOLOv5 and see how it performs on the choosen dataset. 

# Getting started
To execute this repositoy on your local computer, the first step is to clone it. To do so, execute the following code.
```
git clone https://github.com/The-Quantum/insect_recognition.git
```
## Set virtual environment
Once the repository is cloned, navigate into the root dir `insect_recognition` and set the the virtual environment. It must includes the requirement for YOLOv5 as well as packages needed for preprocessiong. I sumerized all those packages into `requirement.txt` file. Personaly, I often combine `pipenv` and `pip` for virtualenv and packages management respectively. It is to be mentioned that one can use `pyenv` to manage multiple `python` versions localy whithout conflictual interactions. Note that most virtual environment managers such as `pipenv` or `virtualen` simply enable to isolated the python environnement of the apps from the local system. This requires that the needed python version to be used in the isolated environment should already by installed localy. However, those virtual environment managers often do not enable installing and managing multiple python versions localy. This is where a multiple python versions manager such as `pyven` come to play. If you have `pipenv` use the following code to set your virtual environment.

```
 cd insect_recognition/           # navigate into the root dir where the virtualenv should be located
 mkdir .venv/                     # create the .venv/ dir to hold the virtualenv
 pipenv -python3.x.x              # this provided python version should be available
 source .venv/bin/activate        # activate the virtualenv
 pip install -r requirement.txt   # install all the required packages
``` 

In case, you do not have `pipenv` fill free to use any other protocol your confortable with to set and activate the virtual environment. Note that one of the simplet way to set a python virtual environment is the python native method which consist running this line of code : `pythonX.X -m venv venv` to create your virtual environment using a specific `pythonX.X` version. To no what python version is available on your system run this code `python --version` or `python3 --version` or `python -V` or `python3 -V`. Then use `source venv/bin/activate` to activate your virtual environment. In case you get trouble creating a virtual environment and can not solve, remove the version specification and run the code again like this`python -m venv venv`. In principle it shoud work.

# Data preparation

## YOLOv5 data format
Let consider the image provide bellow that has 4 distinct objects identified. The YOLOv5 annotation format for each of such object present in the image is given as following :
```
 class center_X center_y width height
```
Obviously, ```class``` refers to the object's class number an has to be an interger while the rest of the annotation refer to center coordinate of the object's bounding box in the image and the width and height of the corresponding bounding box. 

For each distinct object present in a given image of the training set, the corresponding bounding box annotation is saved as a single line in a `.txt` file with the same file name as the image. ***Of course without image extension but .txt instead***. In the case where there are multiple objects in the same image, there is one annotation `.txt` file containing as many lines as there are objects in the image.
<div align="center">YOLOv5 annotation illustration</div>
<p align="center"><img width="800" alt="PR_step1" src="https://github.com/The-Quantum/insect_recognition/blob/main/notebook/Yolo_annotations_illustration.png"></p>

In the example of the figure, the annototion file will containt **4 lines** corresponding to the four objects (2 people, 1 tinnis ball and 1 tennis racket). 

To train `YOLOv5` on a given dataset, the prior requirement is to prepare the annotations of such data set correspondingly to the above described Yolo format. 

## IP102 data format
[IP102] dataset contains more than 75.000 images belonging to 102 categories. About 19.000 images are annotated with bounding boxes for object detection. However, the annotation are save into `.xml` files which are note compatible with YOLOv5. Each `.xml` file prevides information of the bounding boxes containing the insect in the image as well as the insect class and the filename and the size `(width, hight, depth)` of the corresponding image. In case a given image contains many insects, the corresponding `xml` annotation file provides as much bounding box as there are insects.

Bounding boxes are provided in the format ```x_min, y_min, x_main, y_max``` where `(x_min, y_mim)` are the coordinates of the to left corner of the bounding box and `(x_min, y_mim)` that of the bottom right corner. 

Therefore, it is require to write a module that adapt this annotation into YOLOv5 format. This is the purpose of `prepare_annotation.py` module.

To run the annotations module, first download [IP102 v1.1](https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing) the data set and the corresponding annotations. When you follow the link, you have to choose between classification and dectection dataset. Make sure to choose detection data as it is the purpose of this repository.
Then unzip both `JPEGImages.tar` and `Annotations.tar` in the `datasets/`. To do so the simplest way is to copy both file into `datasets/` subdirectory. Then navigates into it and run the following untar code.
```
 tar -xvf Annotations.tar & tar -xvf JPEGImages.tar
```

Then step into the created `Annotations/` dir and run the following code and delete `IP087000986.xml` file.
```
 rm IP087000986.xml
```
The reason for removing this file is that its content organisation is differement from all other files. I decided no to spend time to adapt `prepare_annotation.py` to a single file. Moreover its content present two bounding boxes while the image it is related to contains only one insect.

Now, run the following code from the `insect_recognition/` root directory of this code. 
```
 python prepare_annotation.py
```
You will see the timeline of the reformatation process. This code will create `Yolo_annotation/` folder to located the formated annotations. 
Note that `python prepare_annotation.py` can take six differents arguments including :
- `--data_dir` default value `data_dir`, indicate the root data folder
- `--output_format` default value `multiple`, indicate if formated annotations should be save in a single output file or multiple files
- `--classes_filepath` default value `datasets/classes.txt`, the path to the file tisting the names of classes
- `--output_dir` default value `Yolo_annotation`, the folder to save all annotations files output in case `--output_format` is set to `multiple` 
- `--annot_file` default value `all_annot.txt`, the output file to save all annotations in case `--output_format` is set to `single`.
- `--input_annot_dir`default value `xmls`, the folder name where `.xml` are located. For some dataset, images as well as annatations are spread in different folders. `prepare_annotation.py` is able to handle this situation provided that all annotation folder have the same name and all image folder have the name which should be different from that of annotation folder and include the strint `image` no mater the case `lower` or `upper`.

Hence, the code `python prepare_annotation.py` suppose all the argument are properly set to default options otherwise, the following code shoud be use with proper argument values.
```
 python prepare_annotation.py --data_dir road2020/train --classes_filepath road2020/damages_details_classes.txt --output_dir Yolo_annotation --annot_file Yolo_TF_annotation --output_format multiple
```
The arguments value ara given for illustration and should be modified consequently.

The last think to do concerning the data set should have been to download [`classes.txt`](https://github.com/xpwu95/IP102/blob/master/classes.txt) file which containts the 102 classes of insects. I have done that and you can find this file in `datasets/`.

## TRAIN TEST VAL SPLIT
To train  YOLOv5 the data set have to be split into `train/`, `val/` and `test/` folder. Each of this file folder should contains two subfolders including `images/` and `labels/`. The modules `python train_test_val_split.py` does this properly.
```
 python train_test_val_split.py
```
This will first create all the require folders and subfolders within `dataset/` parent folder. Then, shuffles and splits the images form `JPECImages/` into `train/`, `val/`, `test/` subset in proportion of `0.6, 0.2, 0.2`. Then copy each image or labels of each set to the corresponding `images/` or `labels/` folder.

Note that changing the location of training dataset during data processing is never a good practice as it consumes resources uselessly particularly if training set is in the order of many tens of gigabytes. However, we are conditioned by reqirements of `YOLOv5` for `train`, `test` and `val` data organisation. 

## Run the training
To start the training, navigate into `insect_recognition/yolov5` and run the following code.

```
 python train.py --img 640 --batch 32 --epochs 5\
 --data ../datasets/data.yaml --weights yolov5s.pt --workers 1\
 --project "insect_detection" --name "yolov5s_size640_epochs5_batch32"
```
The option `--img` refers to size of the imput image. In the above code it is set to 640. Therefore, all images will be resized to 640x640 before feeding into the model.`--data` refers to the YAML file contening the data configuration. `--weights` referes to the pretrained YOLOv5 weights. The parameters `--project` and `--name` enable specifying location where the results of the training will be save to. In this specific case the results will be save to `insect_detection/yolov5s_size640_epochs5_batch32`. More details concerning YOLOv5 parameters can be find [here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

The output of the terminal after 10 epochs training is resumed [training_output.txt](https://github.com/The-Quantum/insect_recognition/blob/main/training_output.txt).

## Evaluation
---- TO DO ----
## Run inference
---- TO DO ----
## streamlite or Flask API
---- TO DO ----
