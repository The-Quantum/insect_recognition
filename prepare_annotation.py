import os
import xml.etree.ElementTree as ET
import argparse
import numpy as np
from tqdm import tqdm

class XMLHandler():
    def __init__(self, data_dir: str,
                 output_format :str = "single",
                 Yolo_annotation_dir: str = "Yolo_annotation/",
                 classes_filepath: str = "damage_classes.txt"):
        
        self.method = output_format
        self.data_dir = data_dir
        self.Yolo_annotation_dir = os.path.join(
            data_dir, Yolo_annotation_dir
        )
        
        if (self.method == "multiple") and \
                (not os.path.exists(self.Yolo_annotation_dir)):
            print(f"""Not existing annotation directory.
            It is created at : {self.Yolo_annotation_dir}""")
            os.makedirs(self.Yolo_annotation_dir)
            
        if os.path.expanduser(classes_filepath):
            self.classes = {
                name: idx+1 for idx, name in enumerate(
                  open(classes_filepath).read().splitlines()
                )
            }
        else :
            classes_filepath = os.path.join(
                data_dir, "damage_classes.txt"
            )

            self.classes = {
                name: idx+1 for idx, name in enumerate(
                    open(classes_filepath).read().splitlines()
                )
            }
    
    def parse(self, annotation_filepath):
        
        tree = ET.parse(annotation_filepath)
        
        annotations = {
            "filename": tree.findtext("filename"), 
            "size": {
                "depth":tree.findtext("./size/depth"),
                "height":tree.findtext("./size/height"),
                "width":tree.findtext("./size/width")
            }
        }

        if "." not in annotations["filename"]:
            
            annotations["filename"] = annotations["filename"] + "." + \
                                tree.findtext("path").split(".")[-1]
        
        for index, obj in enumerate(tree.findall("object")):
            
            object_tag = "object" if index < 1 else f"object_{index}"
            
            annotations[object_tag] = {
                "name"  : obj.findtext("name"), 
                "bndbox": {
                    "xmin": obj.findtext("bndbox/xmin"), 
                    "ymin": obj.findtext("bndbox/ymin"),
                    "xmax": obj.findtext("bndbox/xmax"),
                    "ymax": obj.findtext("bndbox/ymax")
                }
            }
            
        return annotations
    
    def to_yolo_fromat(self, annotations):

        annotations_list = []

        object_keys = [
            key for key in annotations.keys() if "object" in key
        ]
        
        if self.method == "single":
            
            img_dirs = [
                x[0] for x in os.walk(self.data_dir) if "image" in x[0].lower()
            ]
            for img_dir in img_dirs:
                img_path = os.path.join(
                    img_dir, annotations["filename"]
                )
                if os.path.exists(img_path):
                    newline = img_path + " "

        elif self.method == "multiple":
            newline = list()

        for key in object_keys:
            bbox = annotations[key]["bndbox"]

            if self.method == "single":
                try :
                    class_id = self.classes[annotations[key]["name"]]
                except KeyError:
                    class_id = 111
                object_annotation = " {},{},{},{},{} ".format(
                    str(float(bbox["xmin"])), str(float(bbox["ymin"])),
                    str(float(bbox["xmax"])), str(float(bbox["ymax"])), class_id
                )
                newline += object_annotation

            elif self.method == "multiple" :
                coords = np.asarray(
                      [float(bbox["xmin"]), float(bbox["ymin"]), 
                       float(bbox["xmax"]), float(bbox["ymax"])])
                coords = self.convert(annotations, coords)

                newline.append(
                    annotations[key]["name"] + " " \
                    + str(coords[0]) + " " + str(coords[1]) + " " \
                    + str(coords[2]) + " " + str(coords[3])
                )
        
        annotations_list.append(newline)
        
        return annotations_list
    
    def convert(self, annotations, coords):
         
        """ Transform Open Images Dataset bounding boxes
            XMin, YMin, XMax, YMax annotaton format into
            the normalized yolo format.
           input :
           -----
           filename_wihtout_extension : str()
              Image file path without .jpg extension. File by default available in Label/ subdir
           coords : np.array()
              OID coordinate of bounding boxes. 
           return :
           ------
           coords : np.array()
              New bounding boxes coordinate in YOLO formats.
        """
        
        coords[2] -= coords[0]
        coords[3] -= coords[1]   

        x_diff = int(coords[2]/2)
        y_diff = int(coords[3]/2)

        coords[0] = coords[0]+x_diff
        coords[1] = coords[1]+y_diff

        image_dim = annotations["size"]

        coords[0] /= float(image_dim["width"])
        coords[1] /= float(image_dim["height"])
        coords[2] /= float(image_dim["width"])
        coords[3] /= float(image_dim["height"])
        
        (coords[0], coords[1], coords[2], coords[3]) = (
            round(coords[0], 4), round(coords[1], 4), 
            round(coords[2], 4), round(coords[3], 4)
        )
        return coords    
    
    def txt_file_path(self, annotation_filepath, annotations):
        file_name = annotation_filepath.split("/")[-1]
        
        if file_name.split(".")[0] == annotations["filename"].split(".")[0]:

            filename_txt = file_name.split(".")[0] + ".txt"

        elif file_name.split(".")[0] != annotations["filename"].split(".")[0]:
            # TO DO : turn this into a warning
            print("There is a problem with the annotation file %s.\
               The annotation filename does not macth provided name %s",\
               file_name, annotations["filename"])
            filename_txt = annotations["filename"].split(".")[0] + ".txt"
            
        else :
            # TO DO : turn this into a warning 
            print(f"Unusual format for the file : {annotations['filename']}")
            filename_txt = file_name.split(".")[0].split("/")[-1] + ".txt"
            
        new_file_path = os.path.join(self.Yolo_annotation_dir, filename_txt)
        
        return new_file_path
    
    def write_file(self, annotation_filepath, annotations, outfileobj = None):

        new_file_path = self.txt_file_path(annotation_filepath, annotations)
        annotations_list = self.to_yolo_fromat(annotations)

        if self.method == "single":

            for line in annotations_list:
                outfileobj.write(line)
                outfileobj.write("\n")

        elif self.method == "multiple":
            #print(annotations_list, new_file_path)
            with open(new_file_path, "w") as outfile:
                for line in annotations_list[0]:
                    outfile.write(line)
                    outfile.write("\n")
        
if (__name__ == "__main__"):

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="datasets/", help='root data directory')
    parser.add_argument('--output_format', type=str, default="multiple", help='output in single or multiple file')
    parser.add_argument('--classes_filepath', type=str, default="datasets/classes.txt", help='class name file')
    parser.add_argument('--output_dir', type=str, default="Yolo_annotation", help='output text directory')
    parser.add_argument('--annot_file', type=str, default="all_annot.txt", help='output file with all annotations')
    parser.add_argument('--input_annot_dir', type=str, default="xmls", help='End input folder name where are annotation files')
    opt = parser.parse_args()

    Preparator = XMLHandler(
        data_dir = opt.data_dir, 
        output_format = opt.output_format, 
        Yolo_annotation_dir = opt.output_dir,
        classes_filepath = opt.classes_filepath
    )
    
    # Identify in datasets/ arborescence all annotation dirs
    annotation_dirs = [
        x[0] for x in os.walk(opt.data_dir) if 
                    opt.input_annot_dir.lower() in x[0].lower()
    ]
    
    # Exclude from the annotation arborescente all the annotations dirs
    #if opt.output_format == "multiple":
    #    annotation_dirs = [
    #        x for x in annotation_dirs if opt.output_dir not in x
    #    ]
    print(annotation_dirs)
    if opt.output_format == "single":
        all_annotations_filepath = os.path.join(
            opt.data_dir, opt.annot_file
        )
        f = open(all_annotations_filepath, "w")

    for dir in tqdm(annotation_dirs):
        
        for file in os.listdir(dir):
            annotation_file_path = os.path.join(dir, file)
            annotations = Preparator.parse(annotation_file_path)
            if "object" not in annotations.keys():
                continue

            if opt.output_format == "single":
                Preparator.write_file(
                    annotation_file_path, annotations, outfileobj=f
                )
            elif opt.output_format == "multiple":
                Preparator.write_file(
                    annotation_file_path, annotations
                )
    
    if opt.output_format == "single":
        f.close() 

# python prepare_annotation.py --data_dir ../rddc2020/yolov5/datasets/road2020/train --classes_filepath ../rddc2020/yolov5/datasets/road2020/damages_details_classes.txt --output_dir Yolo_annotation_1 --annot_file Yolo_TF_annotation --output_format multiple
# python prepare_annotation.py --data_dir ../rddc2020/yolov5/datasets/road2020/train --classes_filepath ../rddc2020/yolov5/datasets/road2020/damages_details_classes.txt --output_dir Yolo_annotation_1 --annot_file Yolo_TF_annotation --output_format single