


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
import json

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
### CONFIGS
##################################################################################


# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("my_dataset_train",)
# cfg.DATASETS.TEST = ("my_dataset_test",)

# cfg.DATALOADER.NUM_WORKERS = 4
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 4
# cfg.SOLVER.BASE_LR = 0.001


# cfg.SOLVER.WARMUP_ITERS = 1000
# cfg.SOLVER.MAX_ITER = 3000 #adjust up if val mAP is still rising, adjust down if overfit
# cfg.SOLVER.STEPS = (1000, 1300, 1800)
# cfg.SOLVER.GAMMA = 0.05

# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 32 + 1 #your number of classes + 1

# cfg.TEST.EVAL_PERIOD = 500


##################################################################################
### INFERENCE WITH D2 SAVED WEIGHTS
##################################################################################


print ("model inference started...")

cfg = get_cfg()
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = os.path.join("/home/appuser/ls_detectron2_objdet/ls_dtron2/models/output", "model_final.pth")
# cfg.DATASETS.TEST = ("my_dataset_test", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.70   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

img_dir = "./test_inf_imgs/"
img_path_list = os.listdir("./test_inf_imgs")
for img_path in img_path_list:
    img = cv2.imread(img_dir + img_path)
    outputs = predictor(img)
    with open("./test_inf_op.txt", "a+") as f:
         f.write(str(outputs))
    print(outputs)

    


