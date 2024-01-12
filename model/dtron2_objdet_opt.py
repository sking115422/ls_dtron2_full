### IMPORTS ###

import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import cv2
import os
import shutil
import json
import math
import time
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances

### FUNCTIONS FOR DATA PREPARATION AND HANDLING ###

def update_img_refs(in_dir):
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
    
    print("updating image references")
  
def coco_train_test_split(in_dir):
    fn = in_dir.split("/")[-1]
    
    if fn == None:
        fn = in_dir.split("/")[-2]
  
    out_dir = os.getcwd() + "/" + fn + "_split"
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

        train_dir = out_dir + "/train"
        os.mkdir(train_dir)
        train_img_dir = train_dir + "/images"
        os.mkdir(train_img_dir)

        test_dir = out_dir + "/test"
        os.mkdir(test_dir)
        test_img_dir = test_dir + "/images"
        os.mkdir(test_img_dir)

        train_split = 0.8

        f = open(in_dir + "/result.json")
        coco_json = json.load(f)
        f.close()

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
            "images": train_img_list,
            "categories": cat_list,
            "annotations": train_ann_list
        }

        test_json = {
            "images": test_img_list,
            "categories": cat_list,
            "annotations": test_ann_list
        }

        train_j_out = json.dumps(train_json, indent=4)
        test_j_out = json.dumps(test_json, indent=4)

        with open(train_dir + "/result.json", "w") as outfile:
            outfile.write(train_j_out)
        with open(test_dir + "/result.json", "w") as outfile:
            outfile.write(test_j_out)
            
        print("creating " + str(train_split) + " train test split to path: " + out_dir)
        
    else:
        print("directory: " + out_dir + " already exists!")
        
def get_soi(str1, start_char, end_char):
    str1 = str(str1)
    offst = len(start_char)
    ind1 = str1.find(start_char)
    ind2 = str1.find(end_char)
    s_str = str1[ind1+offst:ind2]
    return s_str

def createDataDict(fn, outputs):
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
        ann_list.append({"iscrowd": 0, "bbox": bbox_list_new[i], "category_id": class_list_new[i], "bbox_mode": 0})
    
    data_dict = {
        "file_name": fn,
        "height": img_h,
        "width": img_w, 
        "annotations": ann_list
    }
 
    return data_dict

### MAIN FUNCTION ###

def main(input_fn):
    
    # Registering COCO instances
    
    register_coco_instances("my_dataset_train", {}, "./" + input_fn + "_split/train/result.json", "./" + input_fn + "_split/train/images")
    print("training set coco instance registered")
    register_coco_instances("my_dataset_test", {}, "./" + input_fn + "_split/test/result.json", "./" + input_fn + "_split/test/images")
    print("test set coco instance registered")
    
    ### TRAIN CUSTOM D2 DETECTOR ###

    class CocoTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                os.makedirs("coco_eval", exist_ok=True)
                output_folder = "coco_eval"
            return COCOEvaluator(dataset_name, cfg, False, output_folder)

    # Model Configuration Adjustments for Faster Training
    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_test",)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.SOLVER.BASE_LR = 0.0025  # Adjusted for faster convergence
    cfg.SOLVER.MAX_ITER = 1500  # Reduced for faster training
    cfg.SOLVER.STEPS = (500, 1000)
    cfg.SOLVER.GAMMA = 0.1  # Adjusted for faster learning rate decay

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 27 + 1  # Your number of classes + 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 27
    cfg.TEST.EVAL_PERIOD = 250  # More frequent evaluation
    
    ### Looking into balancing the class weights or using focal loss function instead ###

    # Train the Model
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Adjust as per your GPU ids
    print(torch.__version__, torch.cuda.is_available())
    print("clearing GPU memory")
    torch.cuda.empty_cache()
    
    setup_logger()
    
    # Setting paths
    coco_input_base_dir =  "/mnt/nis_lab_research/data/coco_files/aug/"        
    input_fn = "far_shah_1247_v1_all_aug_att"

    # update_img_refs(coco_input_base_dir + input_fn)
    coco_train_test_split(coco_input_base_dir + input_fn) 
    
    launch(
        main,
        num_gpus_per_machine=2,  # Number of GPUs
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:65535",  # Random port; ensure it's free
        args=(input_fn,)
    )


