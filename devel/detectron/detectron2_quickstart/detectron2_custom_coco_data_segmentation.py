# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from pathlib import Path
import os
scratchdir = os.getenv('SCRATCHDIR', ".")
logname = os.getenv('LOGNAME', ".")
from loguru import logger

input_data_dir = Path(scratchdir) / 'data/orig/'
print("input_data_dir =", input_data_dir)
outputdir = Path(scratchdir) / 'data/processed/'
print("outputdir =", outputdir)
logger.debug(f"outputdir={outputdir}")
logger.debug(f"input_data_dir={input_data_dir}")
# print all files in input dir recursively to check everything
logger.debug(str(Path(input_data_dir).glob("**/*")))

from detectron2.data.datasets import register_coco_instances
pathToJson=str(input_data_dir / "coco_training/32/annotations/instances_default.json")
pathToPng=str(input_data_dir / "coco_training/32/images")
print("Json= ",  pathToJson)
print("Png= ",  pathToPng)
register_coco_instances("Parenhyma", {},pathToJson, pathToPng) 

TestJpg=str(input_data_dir / "png-testing/Tx030D_Ven-20220314T115944Z-001/Tx030D_Ven")
register_coco_instances("Parenhyma_Test", {}, pathToJson, pathToPng)
fruits_nuts_metadata = MetadataCatalog.get("Parenhyma")
dataset_dicts = DatasetCatalog.get("Parenhyma")

import random

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    file_path = outputdir/"vis_train"/ Path(d["file_name"]).name
    print("file_path = ", file_path)
    logger.debug(d["file_name"])
    file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(file_path)
    cv2.imwrite(str(file_path), vis.get_image()[:, :, ::-1])




from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file(f"/auto/plzen1/home/bohdany/projects/PigOrganDataSegmentation/devel/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("Parenhyma",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2 #kolik obrazku v 1 okamžik na grafickou kartou
cfg.SOLVER.BASE_LR = 0.02 # jak intenzivně měnime Váhy při backPropagation.
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9 # 4 classes (data, fig, hazelnut)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("Parenhyma_Test", )
predictor = DefaultPredictor(cfg)


from detectron2.utils.visualizer import ColorMode

for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=fruits_nuts_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    file_path = outputdir / "vis_predictions" / Path(d["file_name"]).name
    outputdir.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(file_path), v.get_image()[:, :, ::-1])
    # cv2_imshow(v.get_image()[:, :, ::-1])

