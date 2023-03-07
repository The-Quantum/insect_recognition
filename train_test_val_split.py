import os
import random
import shutil
import tqdm
import argparse
from matplotlib import image
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--test_ratio', type=float, default=.6, help='trainset proportion')
parser.add_argument('--val_ratio', type=float, default=.2, help='testset proportion')
opt = parser.parse_args()

PRESENT_DIR = os.path.dirname(os.path.realpath(__file__))

ANNOTATION_DIR = os.path.join(PRESENT_DIR, "datasets/labels/")
IMAGE_DIR = os.path.join(PRESENT_DIR, "datasets/JPEGImages/")

TRAIN_DIR = os.path.join(PRESENT_DIR, "datasets/train")
TEST_DIR  = os.path.join(PRESENT_DIR, "datasets/test")
VAL_DIR   = os.path.join(PRESENT_DIR, "datasets/val")


IMAGE_LIST = os.listdir(IMAGE_DIR)
SPLIT_IMAGE_NAME = random.shuffle(IMAGE_LIST)
SPLIT_IMAGE_NAME = [filename.split(".")[0] for filename in IMAGE_LIST]

val_ratio  = 0.2
test_ratio = 0.2

train, val, test = np.split(np.array(SPLIT_IMAGE_NAME), 
            [int(len(SPLIT_IMAGE_NAME) * (1 - (opt.val_ratio + opt.test_ratio))),
            int(len(SPLIT_IMAGE_NAME) * (1 - opt.val_ratio))]
        )

print("Train size %s, Val size %s, test size %s 11388 3796 3797", len(train), len(val), len(test))

for dir in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
    image_dir = os.path.join(dir, "images")
    labels_dir = os.path.join(dir, "labels")
    try :
        os.makedirs(image_dir)
    except FileExistsError:
        print("The directory exists : {}".format(image_dir))

    try :
        os.mkdir(labels_dir)
    except FileExistsError:
        print("The directory exists : {}".format(labels_dir))


def displace_files(file_list, img_dest_dir, label_dest_dir):
    print("file displacement in progress ...")
    for item in tqdm(file_list):
        item_img = item + '.jpg'
        item_labels = item + '.txt'
        item_origin_img_path   = os.path.join(IMAGE_DIR, item_img)
        item_origin_label_path = os.path.join(ANNOTATION_DIR, item_labels)

        shutil.copy(item_origin_img_path, img_dest_dir)
        try :
            shutil.copy(item_origin_label_path, label_dest_dir)
        except FileNotFoundError:
            print("Exception FileNotFoundError: No file: {}".format(item_origin_label_path))
            print("The corresponding image is deleted: {}".format(item_origin_img_path))
            os.remove(item_origin_img_path)

        # shutil.move(item_origin_img_path, img_dest_dir)
        # shutil.move(item_origin_label_path, label_dest_dir)

TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, "images")
TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, "labels")
displace_files(train, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)

TEST_IMAGES_DIR = os.path.join(TEST_DIR, "images")
TEST_LABELS_DIR = os.path.join(TEST_DIR, "labels")
displace_files(test, TEST_IMAGES_DIR, TEST_LABELS_DIR)

VAL_IMAGES_DIR = os.path.join(VAL_DIR, "images")
VAL_LABELS_DIR = os.path.join(VAL_DIR, "labels")
displace_files(val, VAL_IMAGES_DIR, VAL_LABELS_DIR)

print("========= THE FILE DISPLACEMENT ENDED SUCCESSFULLY =============")

