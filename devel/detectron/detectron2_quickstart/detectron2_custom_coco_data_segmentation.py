# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
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
pathToJsonTrain=str(input_data_dir / "coco_training/TrainingCoco1/Traning_Coco.json")
pathToJsonTest=str(input_data_dir / "coco_testing/coco_testing1/Testing_Coco3.json")
pathToJsonFinale=str(input_data_dir / "coco_training/26/annotations/instances_default.json")
pathToPngFinale=(input_data_dir / "png-training/Tx026/Tx026D_Ven")
pathToPng=str(input_data_dir / "png_full/PNG2")
print("Json= ",  pathToJsonTrain)
print("Json2= ", pathToJsonTest)
print("Png= ",  pathToPng)
register_coco_instances("Parenhyma", {},pathToJsonTrain, pathToPng) 

TestJpg=str(input_data_dir / "png-testing/Tx030D_Ven-20220314T115944Z-001/Tx030D_Ven")
register_coco_instances("Parenhyma_Test", {}, pathToJsonTest, pathToPng)
register_coco_instances("Parenhyma_Final", {}, pathToJsonFinale, pathToPngFinale)
fruits_nuts_metadata = MetadataCatalog.get("Parenhyma")
dataset_dicts = DatasetCatalog.get("Parenhyma")
dataset_dicts2 = DatasetCatalog.get("Parenhyma_Test") #test na konci 
dataset_dicts3 = DatasetCatalog.get("Parenhyma_Final")
import random

for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    file_path = outputdir/"vis_train"/ Path(d["file_name"]).name
    # print("file_path = ", file_path)
    logger.debug(d["file_name"])
    file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(file_path)
    cv2.imwrite(str(file_path), vis.get_image()[:, :, ::-1])




from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
cfg = get_cfg()
cfg.merge_from_file(f"/auto/vestec1-elixir/home/bohdany/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
print("Parenhyma")
cfg.DATASETS.TRAIN = ("Parenhyma",)
print("Parenhyma-ok")
print("Parenhyma_Test")
cfg.DATASETS.VAL = ("Parenhyma_Test",)
cfg.DATASETS.TEST = ("Parenhyma_Test",)   # no metrics implemented for this dataset
print("Parenhyma_Test-ok")
cfg.DATALOADER.NUM_WORKERS = 2
print("NUM_WORKERS ok")
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
print("model weights ok")
cfg.SOLVER.IMS_PER_BATCH = 5 #kolik obrazku v 1 okamžik na grafickou kartou
print("kolik obrazu ok")
cfg.SOLVER.BASE_LR = 0.00001 # jak intenzivně měnime Váhy při backPropagation.
print("intenyita ok")
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
print("iter ok")
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1   # faster, and good enough for this toy dataset изм
print("mODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE ok")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 # 4 classes (data, fig, hazelnut) изм
print("classes ok")
cfg.OUTPUT_DIR=str(outputdir)
print("outputdir ok")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(" trainer_started ")
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
print(" trainer_finished ")

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
#cfg.DATASETS.TEST = ("Parenhyma_Test", )
print(" test_started ")
predictor = DefaultPredictor(cfg)
print(" test_finished ")
# Ne funguje ta čast

from detectron2.utils.visualizer import ColorMode

for d in dataset_dicts3:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=fruits_nuts_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    print('v', v)
    print(type(v))
    v2 = outputs["instances"].pred_masks.to("cpu").numpy()
   
    file_path = outputdir / "vis_predictions" / Path(d["file_name"]).name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(file_path), v.get_image()[:, :, ::-1])
    
    
    v5 = outputs["instances"].pred_classes.to("cpu").numpy()
    #y=v2[:, :, :]*255
    #y=v2[:, :, :]
    #print(type(y))
    #print(y.shape)
    #print(y)
    print(v5.shape)
    print(v5.size)
    print(v5)
    maskPos=0
    masky={}
    
    for i in v5:
        if i==1:
            
            v6 = outputs["instances"].pred_masks.to("cpu").numpy()
            Maska_org=v6[maskPos, :, :]
            
            if i in masky:
                masky[i]=masky[i]+Maska_org
            else: 
                masky[i]=Maska_org
     
            #data2 = Image.fromarray(Maska.astype(np.uint8))
            #index=str(i)
        
            #file_path2= outputdir / "vis_predictions_mask" /"object_"index"_"Path(d["file_name"]).name
            #file_path2= outputdir / "vis_predictions_mask"/Path(d["file_name"]).name
            #plt.imsave(str(file_path2), data2)
        maskPos=maskPos+1
    for nlabel in masky:
        image=((masky[nlabel]>0).astype(np.uint8))*255
        
        #data2 = Image.fromarray(image.astype(np.uint8))
     
        
        file_path2= outputdir / "vis_predictions_mask"/str(nlabel)/Path(d["file_name"]).with_suffix(".png").name
        file_path2.parent.mkdir(parents=True, exist_ok=True)
        #plt.imsave(str(file_path2), data2)
        cv2.imwrite(str(file_path2), image)
    #cv2.imwrite(str(file_path2), v2[3, :, :]*255)
    # cv2_imshow(v.g[:, :, ::-1])
    
print("all ok")
 #TRY help to pull
