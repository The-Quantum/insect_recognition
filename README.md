# Train YOLOv5 for `Insect pests` recognition

This project is dedicated to the detection and the recognition of insects using YOLOv5. 
The method consist of retraining [YOLOv5](https://github.com/ultralytics/yolov5) on [IP102](https://github.com/xpwu95/IP102) insects pests dataset. 

Insect pests are well known to be a major cause of damage to the agricultural crops with important commercial losses [IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_IP102_A_Large-Scale_Benchmark_Dataset_for_Insect_Pest_Recognition_CVPR_2019_paper.pdf). Our goal here is to retrained `YOLOv5` model on [IP102](https://github.com/xpwu95/IP102) dataset as it has proven state of the art efficiency on object detection datasets in various contexts. We will then evaluate the performances of the newly retrained YOLOv5 and see how it performs in the context of insect detaction and recognition. 

# Getting started
To run this code on your local computer, the first step is to clone the repositoy. To do so, execute the following code.
```
git clone https://github.com/The-Quantum/insect_recognition.git
```
## Set virtual environment
Once the repository is cloned, navigates into the root dir `insect_recognition` and set the virtual environment. 
Activates the virtual environment and installs the require packages. The require packages include also the requirement for YOLOv5. I sumerized all those packages into `requirement.txt` file. I personaly often combine `pipenv` and `pip` for virtualenv settings and packages management. Note that the role of virtual environment managers such as `pipenv` or `venv` simply enable to isolated the python environnement of the apps from the local system. This requires that the needed python version to be used in the isolated environment should already by installed localy. However, those virtual environment managers often do not enable installing and managing multiple python versions localy. This is where a multiple python versions manager such as `pyenv` comes to play to enable installing multiple `python` versions localy whithout conflictual interactions. Details on python versions management are accessible on the [official repository of `pyenv`](https://github.com/pyenv/pyenv)

To create a virtual environment and install the required packages with [`pipenv`](https://pypi.org/project/pipenv/), uses the following code.

```
 cd insect_recognition/            # navigate into the root dir where the virtualenv should be located
 mkdir .venv/                      # create the .venv/ dir to hold the virtualenv
 pipenv --python3.x.x              # this provided python version should be available
 source .venv/bin/activate         # activate the virtualenv. `pipenv shell` can be used as well for activation
 pip install -r requirement.txt    # install all the required packages
``` 

In case, you do not have `pipenv` fill free to use any other method your confortable with to set and activate the virtual environment. Note that one of the simple way to set a python virtual environment is the python native method which consist running this line of code : `pythonX.X -m venv venv` using a specific `pythonX.X` version. To know what python version is available on your system run this code `python --version` or `python3 --version` or `python -V` or `python3 -V`. Then use `source venv/bin/activate` to activate your virtual environment. 

# Data preparation

## YOLOv5 data annotation format
Let consider the image provide bellow that has 4 distinct objects identified. The YOLOv5 annotation format for each of such object present in the image is given as follows:
```
 class center_X center_y width height
```
Obviously, ```class``` refers to the object's class number and has to be an integer while the rest of the annotation refers respectivelly to the center coordinate of the object's bounding box in the image and the width and height of the corresponding bounding box.  

For each distinct object present in a given image of the dataset, the corresponding bounding box annotation is saved as a single line in a `.txt` file with the same file name as the image. ***Of course without image extension but .txt instead***. In the case where there are multiple objects in the same image, there is one annotation `.txt` file containing as many lines as there are objects in the image.

<div align="center">YOLOv5 annotation illustration</div>
<p align="center"><img width="800" alt="PR_step1" src="https://github.com/The-Quantum/insect_recognition/blob/main/notebook/Yolo_annotations_illustration.png"></p>

In the example of the figure, the annototion file will containt **4 lines** corresponding to the four objects (2 people, 1 tennis ball and 1 tennis racket). 

To train `YOLOv5` on a given dataset, the first requirement is to prepare the annotations of the training dataset correspondingly to the above described Yolo format. 

## IP102 data annotation format
[IP102](https://github.com/xpwu95/IP102) dataset contains more than 75.000 images belonging to 102 categories. About 19.000 images are annotated with bounding boxes for object detection. However, the annotation are saved into `.xml` files which are note compatible with YOLOv5. Each `.xml` file prevides information of the bounding boxes containing an insect in the image as well as the correspnding insect class. Also the filename and the size `(width, hight, depth)` of the corresponding image are provided in each annotation file. In case a given image contains many insects, the corresponding `xml` annotation file provides as much bounding box as there are insects.

Bounding boxes are provided in the format ```x_min, y_min, x_main, y_max``` where `(x_min, y_mim)` are the coordinates of the to left corner and `(x_min, y_mim)` that of the bottom right corner. 

Therefore, it is required to write a module that convert these annotations into YOLOv5 format. It is the purpose of `prepare_annotation.py` module.

To run the annotations module, first download [IP102 v1.1](https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing) dataset and the corresponding annotations. When you follow the link, make sure to choose detection data as it is the purpose of this repository.

Then unzip both `JPEGImages.tar` and `Annotations.tar` into `datasets/` folder. To do so, the simplest way is to copy both files into `datasets/`. Then navigates into it and runs the following untar code.
```
 tar -xvf Annotations.tar & tar -xvf JPEGImages.tar
```

Then steps into the created `Annotations/` folder and run the following code to delete `IP087000986.xml` file.
```
 rm IP087000986.xml
```
The reason for removing this file is that its content organisation is differement from all other files. I decided no to spend time to adapt `prepare_annotation.py` to a single file. Moreover its content presents two bounding boxes while the image it is related to contains only one insect.

Next, the `classes.txt` file which containts the 102 classes of insects should be placed in `datasets/`. It can be downloaded from [`classes.txt`](https://github.com/xpwu95/IP102/blob/master/classes.txt).

Now, to prepare annotations in YOLO format, run the following code from the `insect_recognition/` root directory of this code. 
```
 python prepare_annotation.py --output_format multiple --input_annot_dir Yolo_annotation
```
You will see the timeline of the reformatation process. This code will create `Yolo_annotation/` folder to located the formated annotation files. 
Note that `python prepare_annotation.py` can take six differents arguments including :
- `--data_dir` default=`datasets/`, indicates the dataset folder
- `--output_format` default=`multiple`, indicates if the converted annotations should be saved in a `single` output file or `multiple` files
- `--classes_filepath` default=`datasets/classes.txt`, the path to the classe name file
- `--input_annotations_dir` default=`datasets/Annotations`, help='root data directory'
- `--output_dir` default=`Yolo_annotation/`, the folder to save annotation output files in case `--output_format` is set to `multiple` 
- `--annot_file` default=`all_annotations.txt`, the output file to save all annotations in case `--output_format` is set to `single`.
- `--annotation_file_mode` default=`spread`, can take two possible values `spread` or `grouped`. Define if all annotations files are regrouped in a single folder or spread in different folders. Endeed, for some dataset, images as well as annatations are spread in different folders. `prepare_annotation.py` handles both cases. You simply need to provide the correct value of arguments according to the use case. Note that, the option `spread` also works even in case of regrouped files but a bit slower. 
<div align="center">Examples of spread and grouped data</div>
<p align="center"><img width="800" alt="PR_step1" src="https://github.com/The-Quantum/insect_recognition/blob/main/notebook/SPREAD_and_grpup_dataset_organisation.png"></p>

The following code provides a detailed way of running `prepare_annotation.py`. The arguments value are given for illustration and should be modified consequently.
```
 python prepare_annotation.py --data_dir road2020/train --classes_filepath road2020/damages_details_classes.txt --output_dir Yolo_annotation --annot_file Yolo_TF_annotation --output_format multiple
```

## TRAIN TEST VAL SPLIT
To train YOLOv5 the dataset have to be split into `train/`, `val/` and `test/` folders. Each of this folders should contains two subfolders including `images/` and `labels/`. The modules `python train_test_val_split.py` does that properly.
```
 python train_test_val_split.py --test_ratio 0.1 --val_ratio 0.1
```
This will first create all the require folders and subfolders within `dataset/` parent folder. Then, shuffles and splits the images from `JPECImages/` into `train/`, `val/`, `test/` subsets in proportion of `0.6, 0.2, 0.2`. Then, copy each images and labels of each set to the corresponding `images/` and `labels/` subfolders.
The splitting procedure is simply based on random shuflling and does not implement optimized classes distribution in different subset.

Note that changing copying and writting dataset during data processing is never a good practice as it consumes resources uselessly. This will be optimized in future versions.

## Run the training
To start the training, navigate into `insect_recognition/yolov5` and run the following code.

```
 python train.py --img 640 --batch 32 --epochs 5\
 --data ../datasets/data.yaml --weights yolov5s.pt --workers 1\
 --project "insect_detection" --name "yolov5s_size640_epochs5_batch32"
```
The option `--img` refers to the size of the imput image. In the above code it is set to 640. Therefore, all images will be resized to 640x640 before feeding into the model.`--data` refers to the YAML file contening the data configuration. `--weights` referes to the pretrained YOLOv5 weights. The parameters `--project` and `--name` enable specifying location where the results of the training will be save to. In this specific case the results will be save to `insect_detection/yolov5s_size640_epochs5_batch32`. More details concerning YOLOv5 parameters can be find [here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

The output of the terminal after 10 epochs training is resumed [training_output.txt](https://github.com/The-Quantum/insect_recognition/blob/main/training_output.txt).

## Evaluation
---- TO DO ----
## Run inference
---- TO DO ----
## streamlite or Flask API
---- TO DO ----
