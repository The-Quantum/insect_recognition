import os
import re
import xml.sax
import numpy as np
from tqdm import tqdm

PRESENT_DIR  = os.path.dirname(os.path.realpath(__file__))

class XMLHandler(xml.sax.ContentHandler):
   """Handle the parse XML file and extract all annotions
      This annotations incule the filename, the size of the 
      corresponding image, the object name (that is the class name)
      and bounding boxes of each object in the image.
   """
   def __init__(self):
      """ Initialized all parameters needed in anotation dict"""
      self.current_data = ""
      self.filename = ""
      self.name   = ""
      self.width  = ""
      self.height = ""
      self.depth  = ""
      self.xmin   = ""
      self.ymin   = ""
      self.xmax   = ""
      self.ymax   = ""
      self.annotations = {}
      self.increment_object = 0
      
   def startElement(self, tag, attributes):
      """Intialized the corresponding annotation sub dict given the tag name 
      """
      self.current_data = tag
      
      if tag == "size":
        self.annotations["size"] = {}

      elif tag == "object":
        if self.increment_object == 0:
            self.key_name = tag
            self.annotations[self.key_name] = {}
        else :
            self.key_name = tag + "_" + str(self.increment_object)
            self.annotations[self.key_name] = {}

        self.increment_object += 1

      elif tag == "name":
        self.annotations[self.key_name]["name"] = ""

      elif tag == "bndbox":
        self.annotations[self.key_name][tag] = {}
            
   def endElement(self, tag):
      """Check the end of current element and reinitilized 
         self.current element to an empty string
      """
   #  if self.current_data == "width":
   #     print("Width : ", self.width, "CONTENT :", tag)
   #  elif self.current_data == "height":
   #     print("height : ", self.height)
   #  elif self.current_data == "depth":
   #     print("depth : ", self.depth)
   #  elif self.current_data == "ymin":
   #     print("ymin : ", self.ymin)
   #  elif self.current_data == "xmax":
   #     print("xmax : ", self.xmax)
   #  elif self.current_data == "ymax":
   #     print("ymax : ", self.ymax)

   #  if self.current_data == "annotation":
   #    print(parameters)
      self.current_data = ""

   def endDocument(self):
      """Return the annotations at the end of each document"""
      return self.annotations
         
   def characters(self, content):
      """Browser the parsed file and fill self.annotations with 
      the corresponding data
      """
      if self.current_data == "filename":
         self.filename = content
         self.annotations["filename"] = self.filename

      elif self.current_data == "name":
         self.name = content
         self.annotations[self.key_name]["name"] = self.name
         
      elif self.current_data == "xmin":
         self.xmin = content
         self.annotations[self.key_name]["bndbox"]["xmin"] = self.xmin

      elif self.current_data == "ymin":
         self.ymin = content
         self.annotations[self.key_name]["bndbox"]["ymin"] = self.ymin

      elif self.current_data == "xmax":
         self.xmax = content
         self.annotations[self.key_name]["bndbox"]["xmax"] = self.xmax

      elif self.current_data == "ymax":
         self.ymax = content
         self.annotations[self.key_name]["bndbox"]["ymax"] = self.ymax

      elif self.current_data == "width":
         self.width = content
         self.annotations["size"]["width"] = self.width

      elif self.current_data == "height":
         self.height = content
         self.annotations["size"]["height"] = self.height

      elif self.current_data == "depth":
         self.depth = content
         self.annotations["size"]["depth"] = self.depth

def format_annotation(annotations):
   bbox_key = []
   annotations_list = []
   for key in annotations.keys():
      if 'object' in key :
         bbox_key.append(key)
   
         bbox = annotations[key]["bndbox"]
         coords = np.asarray(
                  [float(bbox["xmin"]), float(bbox["ymin"]), 
                     float(bbox["xmax"]), float(bbox["ymax"])])

         coords = convert(annotations, coords)

         newline = annotations[key]["name"] + " " \
                     + str(coords[0]) + " " \
                     + str(coords[1]) + " " \
                     + str(coords[2]) + " " \
                     + str(coords[3])
         
         annotations_list.append(newline)
         
   return annotations_list

def convert(annotations, coords):
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
   #print(coords)
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

   return coords

def remove_first_end_spaces(string):
   return "".join(string.rstrip().lstrip())

def parse_classes():
   classes = {}
   with open("classes.txt", "r") as myFile:
      for num, line in enumerate(myFile, 0):
         line = line.rstrip("\n")
         line = re.sub(r'[0-9]+', '', line)
         line = remove_first_end_spaces(line)
         classes[line] = num
   return classes
   
if (__name__ == "__main__"):

   # creates an XMLReader
   parser = xml.sax.make_parser()

   # turnsoff namepsaces
   parser.setFeature(xml.sax.handler.feature_namespaces, 0)

   # overrides the default Handler
   handler = XMLHandler()
   parser.setContentHandler( handler )

   # step into dataset directory
   DATA_DIR = os.path.join(PRESENT_DIR, "Annotations/")

   for filename in tqdm(os.listdir(DATA_DIR)):
      file_path = os.path.join(DATA_DIR, filename)

      parser.parse(file_path)
      #print(file_path)
      annotations = handler.endDocument()

      annotations_list = format_annotation(annotations)
      
      if filename.split(".")[0] != annotations["filename"].split(".")[0]:
         print(filename, annotations["filename"])
         break
      
      filename_txt  = filename.split(".")[0] + ".txt"
      new_file_path = os.path.join(PRESENT_DIR, "labels/", filename_txt)

      with open(new_file_path, "w") as outfile:
         for line in annotations_list:
            outfile.write(line)
            outfile.write("\n")