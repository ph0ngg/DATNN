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
import traceback

from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import logging
import os
from scipy.spatial.distance import cosine, cdist

import sys
import time
from collections import OrderedDict
from ultralytics import YOLO
import tqdm

import torch
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import random

sys.path.append('./folder')

from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader, build_reid_train_loader
#from fastreid.evaluation.testing import flatten_results_dict
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

sys.path.append('./sort')
from sort import Sort, KalmanBoxTracker

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def person_embedding(model, imgs):
    imgs = np.expand_dims(imgs, axis= 0)
    imgs = np.transpose(imgs, (0, 3, 1, 2)) #(B, C, H, W)
    imgs = torch.from_numpy(imgs)
    imgs = imgs.type(torch.FloatTensor)  
    imgs = imgs.cuda()
    person_emb = model(imgs)
    person_emb = F.normalize(person_emb)
    person_emb = person_emb.cpu().data.numpy()
    # view_prop = model.backbone(imgs)[1].detach().cpu().numpy()
    # view = np.argmax(view_prop)
    return person_emb#, view

def crop_img(img, x1, y1, x2, y2):
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    cropped_img = img[int(y1):int(y2), int(x1):int(x2)] #H, W
    cropped_img = cv2.resize(cropped_img, (64, 128))

    return cropped_img

def view_predict(kpt):
        view = 0
        if kpt[0] < 0.5:
            view = 2
        else:
            kpt_distance = np.sqrt((kpt[15]-kpt[18])**2 + (kpt[16]-kpt[19])**2) + np.sqrt((kpt[33]-kpt[36])**2 + (kpt[34] - kpt[37])**2)
            if kpt_distance < 0.45:
                view = 1
            else:
                view = 0
        return view

def score(bbox, kpt):
    conf = 0
    for i in range(2, len(kpt), 3):
        conf += kpt[i]
    bbox_size_score = bbox.shape[0] * bbox.shape[1] / (128*64)
    bbox_ratio_score = (bbox.shape[1]/bbox.shape[0]) / 2
    return bbox_ratio_score*bbox_size_score + conf

def tracking(path):
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    emb_model = build_model(cfg)
    MONGO_DB = "mydb"

    #Creating a pymongo client
    #url = "mongodb://{}:{}@{}:{}/{}?authSource=admin".format(MONGO_USER, MONGO_PASS, MONGO_HOST, MONGO_PORT, MONGO_DB)
    url = 'mongodb://localhost:27017/'
    client = MongoClient(url)
    db = client[MONGO_DB]
    rt_tracking = {}
    state_dict = torch.load('./folder/tools/khongconv5_lossview0,5.pth')['model']

    emb_model.backbone.load_state_dict(state_dict, strict = False)#['model'])
    emb_model.eval()
    detect_kpt_model = YOLO('yolov8s-pose.pt')
    mot_trackers = []
    track_prev = []
    emb = {}
    start_times = {}
    count = 0
    txt_folder = './FastAPI1/result/result_txt'
    videos = []
    txt_files = []
    current_id = 0
    for video_name in sorted(os.listdir(path)):
        video_path = os.path.join(path, video_name)
        txt_file = os.path.join(txt_folder, video_name + '.txt')
        txt_files.append(txt_file)
        tmp_video = cv2.VideoCapture(video_path)
        videos.append(tmp_video)
        mot_tracker = Sort()
        mot_trackers.append(mot_tracker)
    files = [open(path, 'w') for path in txt_files]
    flag = True
    time1 = time.time()
    while flag:
        if count%3 ==0:
            for i, video in enumerate(videos):
                ret, img_show = video.read()
                if img_show is None:
                    flag = False
            count +=1 
            continue
        for i, video in enumerate(videos):
            ret, img_show = video.read()
            if img_show is None:
                flag = False
            preds = detect_kpt_model(img_show, verbose = False)
            for r in preds:
              try:
                xy_bboxs = r.boxes.xyxy.cpu().numpy()
                conf_bboxs = r.boxes.conf.cpu().numpy()
                xy_keypoints = r.keypoints.xyn.cpu().numpy()
                conf_keypoints = r.keypoints.conf.cpu().numpy()
                num_people = conf_bboxs.shape
                zeros = np.zeros((conf_bboxs.shape[0], 1))
                keypoints = np.concatenate((xy_keypoints, np.expand_dims(conf_keypoints, axis = 2)), axis = 2)
                keypoints = keypoints.reshape(keypoints.shape[0], -1) #shape: num_people, 51
                bboxs = np.concatenate((xy_bboxs, np.expand_dims(conf_bboxs, axis = 1)), axis = 1)
                res = np.concatenate((bboxs ,zeros, keypoints), axis = 1)
                track_bbs_ids = mot_trackers[i].update(res)
                same_view = True  
                dist = 1.01
                threshold = .25
                min_dist = 1
                this_same_view = True
# #     --------------------------------------------------------------
#              #   mot_tracker: Danh sách các track trong frame hiện tại
                id_tracking = []
                for tracker in mot_trackers[i].trackers:
                    id_tracking.append(tracker.id)
                for tracker in mot_trackers[i].trackers:
                    bbox = tracker.get_state()[0]  # x1, y1, x2, y2
                    bbox_img = crop_img(img_show, bbox[0], bbox[1], bbox[2], bbox[3])
                    vector = list(person_embedding(emb_model, bbox_img)[0])
                    view = view_predict(tracker.kpt)
                    vector.append(view)

                    for idx in range(len(vector)):
                        vector[idx] = float(vector[idx])

                    if tracker.is_tracking and not tracker.last_frame_is_tracking:  # track moi
                        this_gid = -1
                        this_same_view = False

                        for track_id, track in rt_tracking.items():
                            track_vt = track['embedding_vector']  # [n, 1536]
                            vt = [row[:-1] for row in track_vt]

                            # Kiểm tra xem có id của global track trong các track đang xét chưa
                            if track_id in id_tracking:
                                continue
                            
                            # Nếu chưa thì kiểm tra khoảng cách của track đang xét và global track
                            cosine_matrix = cdist(vt, np.expand_dims(np.array(vector[:-1]), axis=0), 'cosine')  # [n, 1]
                            dist = np.min(cosine_matrix)
                            index = np.unravel_index(np.argmin(cosine_matrix), cosine_matrix.shape)
                            same_view = track_vt[index[0]][-1] == view

                            if dist < min_dist:
                                min_dist = dist
                                this_gid = track_id
                                this_same_view = same_view

                        if this_same_view:
                            threshold = 0.42
                        else:
                            threshold = 0.6

                        if min_dist < threshold:
                            tracker.id = this_gid
                        else:  # track mới chưa xuất hiện trong db
                            tracker.id = current_id
                            current_id += 1

                    if tracker.id not in rt_tracking:
                        rt_tracking[tracker.id] = {'embedding_vector': []}

                    # Update the dictionary instead of the database
                    rt_tracking[tracker.id]['embedding_vector'].append(vector)
        
        # Ensure the list only keeps the last 50 vectors
                    if len(rt_tracking[tracker.id]['embedding_vector']) > 50:
                        rt_tracking[tracker.id]['embedding_vector'] = rt_tracking[tracker.id]['embedding_vector'][-50:]
                for j in range(len(track_bbs_ids.tolist())):
                    coords = track_bbs_ids.tolist()[j]
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            #         #print(coords[4])
                    name_idx = int(coords[4])-1
            #         name = 'ID: {}'.format(str(name_idx))
            #         color1 = (255, 0, 0)
            #         color2 = (0, 0, 255)
            #         random.seed(int(coords[4]))
            #         color = (255*random.random(), 255*random.random(), 255*random.random())
                    width = x2 - x1
                    height = y2 - y1
                    files[i].write(str(count) + ', ' +str(name_idx) +', '+ str(x1) + ', '+ str(y1) + ', ' + str(width) + ', '+ str(height)+ ', 1, 1, 1')
                    files[i].write('\n')
            #   #      ----------------------------------------------------------------
            #         kpt = keypoints[j]
            #         cv2.rectangle(img_show, (x1, y1), (x2, y2), color, 1)
            #         cv2.putText(img_show, name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            #         cv2.putText(img_show, str(round(conf_bboxs[j], 2)), (x2, y2-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
              except Exception as e:
                if str(e) != "'NoneType' object has no attribute 'cpu'":
                    traceback.print_exc()

                continue
        count += 1
        print(count)

    for video in videos:
        video.release()
    for file in files:
        file.close()
    time2 = time.time()
    print('FPS: ', count/(time2-time1))
tracking(r'./FastAPI1/upload_folder')