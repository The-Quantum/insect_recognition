import re
from torch import classes
import yaml

def remove_first_end_spaces(string):
   return "".join(string.rstrip().lstrip())

def parse_classes():
   classes = {}
   with open("datasets/classes.txt", "r") as myFile:
      for num, line in enumerate(myFile, 0):
         line = line.rstrip("\n")
         line = re.sub(r'[0-9]+', '', line)
         line = remove_first_end_spaces(line)
         classes[line] = num
   return classes

classes = parse_classes()
classes_list = list(classes.keys())

config = {'path': '/home/rd-besttic/Documents/project/github_project/insect_recognition/datasets',
          'train': '/home/rd-besttic/Documents/project/github_project/insect_recognition/datasets/train',
          'val': '/home/rd-besttic/Documents/project/github_project/insect_recognition/datasets/val',
          'nc': len(classes_list),
          'names': classes_list}
 
with open("datasets/data.yaml", "w") as file:
   yaml.dump(config, file, default_flow_style=False)
#python train.py --img 640 --batch 16 --epochs 5 --data ../datasets/data.yaml --weights yolov5s.pt --workers 1 --project "insect_detection" --name "yolov5s_size640_epochs5_batch16"