


##################################################################################
### IMPORTS
##################################################################################


# install dependencies: (use cu101 because colab has CUDA 10.1)
# opencv is pre-installed on colab
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

import gc
#del variables
gc.collect()
# Gong added this:
torch.cuda.empty_cache()
#torch.cuda.memory_summary(device=None, abbreviated=False)

# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import json
import math
import os
import shutil
import re

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

#from .detectron2.tools.train_net import Trainer
#from detectron2.engine import DefaultTrainer
# select from modelzoo here: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#coco-object-detection-baselines

from detectron2.config import get_cfg
#from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os

from detectron2.utils.visualizer import ColorMode
import glob

import random
from detectron2.utils.visualizer import Visualizer
import numpy as np 
import os


##################################################################################
### IMPORT AND REGISTER CUSTOM D2 DATA
##################################################################################


def update_img_refs (in_dir):

    f = open(in_dir + "/result.json")
    coco_json = json.load(f)
    f.close()

    img_list = coco_json["images"]

    for i in range(0, len(img_list)):
        img_name = img_list[i]["file_name"].split("/")[-1]
        new_path = "./" + img_name
        coco_json["images"][i]["file_name"] = new_path

    j_out = json.dumps(coco_json, indent=4)

    os.remove(in_dir + "/result.json")

    f = open(in_dir + "/result.json", "w")
    f.write(j_out)
    f.close()
    
    print ("updating image references")
  
def coco_train_test_split (in_dir):
    
    fn = in_dir.split("/")[-1]
    
    if fn == None:
        fn = in_dir.split("/")[-2]
  
    out_dir = os.getcwd() + "/" + fn + "_split"
    
    if not os.path.exists(out_dir):

        os.mkdir(out_dir)

        train_dir = out_dir + "/train"
        os.mkdir(out_dir + "/train")
        train_img_dir = train_dir + "/images"
        os.mkdir(train_img_dir)

        test_dir = out_dir + "/test"
        os.mkdir(out_dir + "/test")
        test_img_dir = test_dir + "/images"
        os.mkdir(test_img_dir)

        train_split = 0.8

        f = open (in_dir + "/result.json")

        coco_json = json.load(f)

        num_img = len(coco_json["images"])

        img_list = coco_json["images"]
        cat_list = coco_json["categories"]
        ann_list = coco_json["annotations"]

        train_num = math.floor(num_img * train_split)

        train_img_list = img_list[0:train_num]
        test_img_list = img_list[train_num:]

        for each in train_img_list:
            img_name = each["file_name"].split("/")[-1]
            shutil.copy(in_dir + "/images/" + img_name, train_img_dir + "/" + img_name)

        for each in test_img_list:
            img_name = each["file_name"].split("/")[-1]
            shutil.copy(in_dir + "/images/" + img_name, test_img_dir + "/" + img_name)

        co_val = train_img_list[-1]["id"]

        train_ann_list = [] 
        test_ann_list = []

        for each in ann_list:

            if each["image_id"] <= co_val:
                train_ann_list.append(each)
            else:
                test_ann_list.append(each)

        train_json = {
            "images" : train_img_list,
            "categories": cat_list,
            "annotations":train_ann_list
        }

        test_json = {
            "images" : test_img_list,
            "categories": cat_list,
            "annotations":test_ann_list
        }

        train_j_out = json.dumps(train_json, indent=4)
        test_j_out = json.dumps(test_json, indent=4)

        with open(train_dir + "/result.json", "w") as outfile:
            outfile.write(train_j_out)
        with open(test_dir + "/result.json", "w") as outfile:
            outfile.write(test_j_out)
            
        print ("creating " + str(train_split) + " train test split to path: " + out_dir)
        
    else:
        
        print("file " + out_dir + " already exists!")
        
# Setting paths

coco_input_base_dir =  "./../coco_files/"        
input_fn = "coco_prelab_743_n"

update_img_refs(coco_input_base_dir + input_fn)
coco_train_test_split(coco_input_base_dir + input_fn) 
    
register_coco_instances("my_dataset_train", {}, "./" + input_fn + "_split/train/result.json","./" + input_fn + "_split/train/images")
print("training set coco instance registered")
register_coco_instances("my_dataset_test", {}, "./" + input_fn + "_split/test/result.json","./" + input_fn + "_split/test/images")
print("test set coco instance registered")



##################################################################################
### TRAIN CUSTOM D2 DETECTOR
##################################################################################


# We are importing our own Trainer Module here to use the COCO validation evaluation during training. Otherwise no validation eval occurs.

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


# Setting model configs

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001

cfg.SOLVER.MAX_ITER = 20

### OG
# cfg.SOLVER.WARMUP_ITERS = 1000
# cfg.SOLVER.MAX_ITER = 3000 #adjust up if val mAP is still rising, adjust down if overfit
# cfg.SOLVER.STEPS = (1000, 1300, 1800)
# cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 32 + 1 #your number of classes + 1

### OG
# cfg.TEST.EVAL_PERIOD = 500


# Training the model

print ("model training started...")

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# Testing the model

print ("model testing started...")

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
inference_on_dataset(trainer.model, val_loader, evaluator)


##################################################################################
### INFERENCE WITH D2 SAVED WEIGHTS
##################################################################################


def get_soi(str1, start_char, end_char):
    str1 = str(str1)
    offst = len(start_char)
    ind1 = str1.find(start_char)
    ind2 = str1.find(end_char)
    s_str = str1[ind1+offst:ind2]
    return s_str

def createDataDict (fn, outputs):
	img_shape = list(outputs["instances"].image_size)
	img_h = int(img_shape[0])
	img_w = int(img_shape[1])

	class_list = get_soi(outputs["instances"].pred_classes, "[", "]").split(",")
	class_list_new = []
	for each in class_list:
		class_list_new.append(int(each.strip()))

	bbox_list = get_soi(outputs["instances"].pred_boxes, "[[", "]]").split("]")
	bbox_list_new = []
	for each in bbox_list:
		bbox = re.sub("['[,\n]", "", each).split(" ")
		bbox_new = []
		for item in bbox:
			if item != "":
				bbox_new.append(float(item))
		bbox_list_new.append(bbox_new)

	ann_list = []
	for i in range(0, len(class_list)):
		# og was "bbox_mode": "<BoxMode.XYWH_ABS: 1>"
		ann_list.append({"iscrowd": 0, "bbox": bbox_list_new[i], "category_id": class_list_new[i], "bbox_mode": 0})
	
	data_dict = {
		"file_name": fn,
		"height": img_h,
		"width": img_w, 
		"annotations": ann_list
	}
 
	return data_dict


### File paths

img_out_dir = "./img_out/"
# img_in_dir = "./test_inf_imgs"
img_in_dir = "./" + input_fn + "_split/test/images/"
results_dir = "./results"

if not os.path.exists(img_out_dir):
    os.makedirs(img_out_dir)
    
if not os.path.exists(img_in_dir):
    os.makedirs(img_in_dir)
    
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
    

print ("model inference started...")



### Starting inference

print ("model inference started...")
cfg.MODEL.WEIGHTS = os.path.join("./old_output/output_500", "model_final.pth")
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.70
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
predictor = DefaultPredictor(cfg)

img_path_list = os.listdir(img_in_dir)

master_dict = []

for img_path in img_path_list:
	img = cv2.imread(img_in_dir + img_path)
	outputs = predictor(img)
	if outputs["instances"].__len__() > 0:
		print(outputs)
		data_dict = createDataDict(img_in_dir + img_path, outputs)
		vis = Visualizer(img[:, :, ::-1], scale=1)
		out = vis.draw_dataset_dict(data_dict)
		cv2.imwrite("./img_out/"+img_path, out.get_image()[:, :, ::-1])
		master_dict.append(data_dict)
		with open("./results/data_dict.json", "w+") as f:
			f.write(json.dumps(master_dict))
	else:
		print("model inference has detected no elements of interest... so img will be skipped.")
