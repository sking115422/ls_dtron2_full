import os
import shutil

def delDir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        
delDir("coco_eval")

pn = os.listdir("./")
for each in pn:
    if each.split("_")[-1] == "split":
        delDir(each)

delDir("img_out")
delDir("output")
delDir("results")

