##################################################################################
### IMPORTS
##################################################################################

import torch, torchvision
import numpy as np
import cv2
import os
import json
import re
import glob
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use two GPUs

# Verify CUDA availability and number of GPUs
print(torch.__version__, torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())

# Clear GPU cache
torch.cuda.empty_cache()

# Setup logger for Detectron2
setup_logger()

##################################################################################
### INFERENCE WITH D2 SAVED WEIGHTS
##################################################################################

img_out_dir = "./img_out/"
img_in_dir = "/home/dtron2_user/ls_dtron2_full/model/far_rev_708_coco_bal_split/test/images/"
results_dir = "./results"

if not os.path.exists(img_out_dir):
    os.makedirs(img_out_dir)
    
if not os.path.exists(img_in_dir):
    os.makedirs(img_in_dir)
    
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print ("model inference started...")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join("/home/dtron2_user/ls_dtron2_full/model/old_output/output_708_bal/", "model_final.pth")
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 32 + 1 #your number of classes + 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 27
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
predictor = DefaultPredictor(cfg)

img_path_list = os.listdir(img_in_dir)

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

    


