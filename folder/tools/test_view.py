# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import re
import copy
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import urllib.parse

from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import logging
import os
import sys
import time
from collections import OrderedDict
from ultralytics import YOLO
import tqdm

import torch
import cv2
import numpy as np
import random

from torch.nn.parallel import DistributedDataParallel

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader, build_reid_train_loader
from fastreid.evaluation.testing import flatten_results_dict
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.modeling import build_model
from fastreid.solver import build_lr_scheduler, build_optimizer
from fastreid.evaluation import inference_on_dataset, print_csv_format, ReidEvaluator
from fastreid.utils.checkpoint import Checkpointer, PeriodicCheckpointer
from fastreid.utils import comm
from fastreid.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter
)

logger = logging.getLogger("fastreid")

sys.path.append('/home/phongnn/test/test/sort')
from sort import Sort, KalmanBoxTracker

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg



def person_embbeding(model, img):
    # img = img/255
    # img[:, :, 0] = img[:, :, 0] - 0.485
    # img[:, :, 1] = img[:, :, 1] - 0.456
    # img[:, :, 2] = img[:, :, 2] - 0.406
    # img[:, :, 0] /= 0.229
    # img[:, :, 1] /= 0.224
    # img[:, :, 2] /= 0.225
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis= 0)
    img = torch.from_numpy(img)
    img = img.type(torch.FloatTensor).cuda()
    
    # feature = model.backbone(img)[0]
    # person_emb = model.heads(feature).detach().cpu().numpy()
    person_emb = model(img).detach().cpu().numpy()
    view_prop = model.backbone(img)[1].detach().cpu().numpy()
    view = np.argmax(view_prop)
    return person_emb, view

def main(args):
    cfg = setup(args)
    emb_model = build_model(cfg)
    MONGO_HOST = "192.168.1.220"
    MONGO_PORT = 27017
    MONGO_DB = "mydb"
    MONGO_USER = "awlvn"
    MONGO_PASS = urllib.parse.quote_plus("123456a@")

    #Creating a pymongo client
    url = "mongodb://{}:{}@{}:{}/{}?authSource=admin".format(MONGO_USER, MONGO_PASS, MONGO_HOST, MONGO_PORT, MONGO_DB)
    client = MongoClient(url)
    db = client[MONGO_DB]
    cluster_centers = db['cluster_centers']
    emb_db = db['new_emb_vector']
    state_dict = torch.load('/home/phongnn/test/test/fast-reid/tools/model_loss/khongconv5_lossview0,5.pth')

    emb_model.load_state_dict(state_dict['model'])
    emb_model.eval()
    path = '/home/phongnn/test/test/Market1203/Market1203_orient_3classes_split/train'
    model = YOLO('yolov8m-pose.pt')
    a = []
    count = 0
    count_true=  0
    # for folder in os.listdir(path):
    #     folder_path = os.path.join(path, folder)
    #     if folder == '1_side': 
    #         for img_path in sorted(os.listdir(folder_path)):
    #           try:
    #img = cv2.imread(os.path.join(folder_path, img_path))
    img = cv2.imread('/home/phongnn/test/test/AIC23/train/S002/query/9_c1s2_6910_00.jpg')
    pred = model(img, verbose= False)
    for r in pred:
        xy_keypoints = r.keypoints.xyn.cpu().numpy()
        conf_keypoints = r.keypoints.conf.cpu().numpy()
        if conf_keypoints[0][0] > 0.5:
            distance = np.sqrt((xy_keypoints[0][5][0]-xy_keypoints[0][6][0])**2 + (xy_keypoints[0][5][1]-xy_keypoints[0][6][1])**2) +\
                        np.sqrt((xy_keypoints[0][11][0]-xy_keypoints[0][12][0])**2 + (xy_keypoints[0][11][1]-xy_keypoints[0][12][1])**2)
            # if distance < 0.45:
            #    count_true += 1
            print(distance)
        else:
            print('back')
    #             count += 1
    #           except:
    #             count += 1
    #             continue          
    # print(count_true/count)    

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
