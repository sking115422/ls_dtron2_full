##################################################################################
### IMPORTS
##################################################################################

import torch, torchvision
import numpy as np
import cv2
import os
import shutil
import json
import re
import glob
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from multiprocessing import Process, Manager

##################################################################################
### FUNCTIONS
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
        if each.strip().isdigit():
            class_list_new.append(int(each.strip()))
        else:
            print(f"Invalid class ID: {each}")

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


def inference_on_gpu(gpu_id, img_paths, cfg):
    torch.cuda.set_device(gpu_id)
    predictor = DefaultPredictor(cfg)
    
    master_dict = []
    
    for img_path in img_paths:
        
        # Creating master dictionary of detected elements
        img = cv2.imread(img_path)
        outputs = predictor(img)
        data_dict = createDataDict(img_path, outputs)
        master_dict.append(data_dict)
        
        # Drawing bounding boxes for each identified element on img
        if outputs["instances"].__len__() > 0:
            print(outputs)
            vis = Visualizer(img[:, :, ::-1], scale=1)
            out = vis.draw_dataset_dict(data_dict)
            output_filename = os.path.join(img_out_dir, os.path.basename(img_path))
            cv2.imwrite(output_filename, out.get_image()[:, :, ::-1])
            
            # Save the results
            with open(os.path.join(results_dir, "tmp", f"data_dict_{gpu_id}.json"), "w+") as f:
                f.write(json.dumps(master_dict))


def split_processing(img_path_list, cfg):
    gpu_count = 2  # Number of GPUs
    full_splits = len(img_path_list) // gpu_count
    remainder = len(img_path_list) % gpu_count
    split_datasets = [img_path_list[:full_splits + remainder]]
    split_datasets += [img_path_list[i:i + full_splits] for i in range(full_splits + remainder, len(img_path_list), full_splits)]
    
    processes = []
    
    for i, dataset in enumerate(split_datasets):
        p = Process(target=inference_on_gpu, args=(i, dataset, cfg))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

##################################################################################
### MAIN
##################################################################################

def main():

    # Inference
    print("Model inference started...")
    img_path_list = [os.path.join(img_in_dir, p) for p in os.listdir(img_in_dir)]
    split_processing(img_path_list, cfg)
    
    # Combining output from each process then writing it back to master file
    res_list = []
    res_path = os.listdir(os.path.join(results_dir, "tmp"))
    for res in res_path:
        fp = os.path.join(os.path.join(results_dir, "tmp"), res)
        with open(fp, "r") as f:
            res_list.append(json.load(f))
    res_list = [item for sublist in res_list for item in sublist]
    
    # Save the results
    with open(os.path.join(results_dir, "data_dict.json"), "w+") as f:
        f.write(json.dumps(res_list))
    print("Inference completed.")
    
    # Clean up tmp directory
    if os.path.exists(os.path.join(results_dir, "tmp")):
        shutil.rmtree(os.path.join(results_dir, "tmp"))

##################################################################################
### RUNNING MAIN - CONFIG & PATH
##################################################################################

if __name__ == '__main__':
    
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join("/home/dtron2_user/ls_dtron2_full/model/output", "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 27
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25

    # Paths
    img_out_dir = "./img_out/"
    # img_in_dir = "/home/dtron2_user/ls_dtron2_full/model/far_rev_708_coco_bal_split/test/images/"
    img_in_dir = "/home/dtron2_user/ls_dtron2_full/model/far_shah_1247_v1_bal_split/test/images/"
    results_dir = "./results/"
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Creating tmp directory for output for each process
    os.makedirs(os.path.join(results_dir, "tmp"), exist_ok=True)
    
    main()

