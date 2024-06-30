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

def view_predict(kpts):
    views = []
    for kpt in kpts:
        view = 0
        if kpt[0] < 0.5:
            view = 2
        else:
            kpt_distance = np.sqrt((kpt[15]-kpt[18])**2 + (kpt[16]-kpt[19])**2) + np.sqrt((kpt[33]-kpt[36])**2 + (kpt[34] - kpt[37])**2)
            if kpt_distance < 0.45:
                view = 1
            else:
                view = 0
        views.append(float(view))
    return views

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
    emb_db = db['embedding']
    state_dict = torch.load('./folder/tools/khongconv5_lossview0,5.pth')['model']

    emb_model.backbone.load_state_dict(state_dict, strict = False)#['model'])
    emb_model.eval()
    detect_kpt_model = YOLO('yolov8s-pose.pt')
    mot_tracker = Sort()
    track_prev = []
    emb = {}
    start_times = {}
    count = 0
    txt_folder = './result/result_txt'

    for filename in sorted((path)):
        print(filename)
        video_path = filename
        video_name = filename.split('\\')[-1][:-4]
        txt_file = os.path.join(txt_folder, video_name + '.txt')
        video = cv2.VideoCapture(video_path)
        f = open(txt_file, 'w')
        while (video.isOpened()):
            ret, img_show = video.read()
            if img_show is None:
                for track in track_prev:
                    try:
                        save_img = emb[track.id][:10] #chọn 10 ảnh có kích cỡ to nhất/
                        emb_doc = []
                        bboxs, kpts = zip(*[(x[0], x[1]) for x in save_img])
                        views = np.array(view_predict(kpts))
                        views_reshape = np.reshape(views, (-1, 1))
                        emb_vectors = list(person_embedding(emb_model, bboxs))
                        emb_vectors = np.concatenate((emb_vectors, views_reshape), axis = 1)
                        emb_vectors = list(emb_vectors)
                        for m in range(len(emb_vectors)):
                            emb_vectors[m] = list(emb_vectors[m])
                            for n in range(len(emb_vectors[m])):
                                emb_vectors[m][n] = float(emb_vectors[m][n])
                        doc1 = {"key": track.id, "embedding_vector": emb_vectors, "start_time": start_times[track.id], "end_time": count}
                        #doc1 = {f"{person_id}" : emb_doc}
                        emb_db.insert_one(doc1)
                        emb.pop(track.id)
                    except:
                        continue
                break
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
                track_bbs_ids = mot_tracker.update(res)   
# #     --------------------------------------------------------------
#              #   mot_tracker: Danh sách các track trong frame hiện tại
                for tracker in mot_tracker.trackers:
                    bbox = tracker.get_state()[0] #x1, y1, x2, y2
                    start_time = count
                    if (1.5*abs(bbox[2]-bbox[0]) <= abs(bbox[3]-bbox[1])) & abs(bbox[2]-bbox[0] >0) & abs(bbox[3]-bbox[1] >0): # chỉ lưu các box có chiều cao > 2 lần chiều rộng  
                        new_box =  (crop_img(img_show, bbox[0], bbox[1], bbox[2], bbox[3]), tracker.kpt)
                        if tracker.id not in emb:
                            emb[tracker.id] = [new_box]
                        else:
                            # random_number = random.random()
                            # if random_number > 0:  
                                if len(emb[tracker.id]) < 10:                                             
                                    emb[tracker.id].append(new_box) #luu lai 10 box của mỗi id
                                else:
                                    emb[tracker.id].sort(key= lambda x: score(x[0], x[1]), reverse = True)  #sort các box
                                    if score(emb[tracker.id][-1][0], emb[tracker.id][-1][1]) < score(new_box[0], new_box[1]): #so sánh score box cuối cùng với box mới
                                        emb[tracker.id].pop(-1)
                                        emb[tracker.id][-1] = new_box
                    if tracker.is_tracking:
                        if tracker.id not in start_times:
                            start_times[tracker.id] = start_time

                    if tracker.last_frame_is_tracking and not tracker.is_tracking:
                        try:
                            save_img = emb[tracker.id][:10] #chọn 10 ảnh có kích cỡ to nhất/
                            print("successful")
                        except:
                            continue
                        emb_doc = []
                        bboxs, kpts = zip(*[(x[0], x[1]) for x in save_img])
                        views = np.array(view_predict(kpts))
                        views_reshape = np.reshape(views, (-1, 1))
                        emb_vectors = list(person_embedding(emb_model, bboxs))
                        print(np.array(emb_vectors).shape)
                        emb_vectors = np.concatenate((emb_vectors, views_reshape), axis = 1)
                        emb_vectors = list(emb_vectors)
                        for m in range(len(emb_vectors)):
                            emb_vectors[m] = list(emb_vectors[m])
                            for n in range(len(emb_vectors[m])):
                                emb_vectors[m][n] = float(emb_vectors[m][n])
                        doc1 = {"key": tracker.id, "embedding_vector": emb_vectors, "start_time": start_times[tracker.id], "end_time": count-1}
                        # for i, img in enumerate(bboxs):
                            # cv2.imwrite('/home/phongnn/test/test/EPFL/result_reid/track_img/' + str(tracker.id) + '_' + str(i) + '.jpg', img)
                        #doc1 = {f"{person_id}" : emb_doc}
                        query = {'key': tracker.id}
                        emb_db.find_one_and_replace(query, doc1, upsert=True)
                num_people = len(track_bbs_ids.tolist()) 
                coors_later, confs_later, coors_prev, confs_prev = [[]]*num_people, [[]]*num_people, [[]]*num_people, [[]]*num_people
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
                    f.write(str(count) + ', ' +str(name_idx) +', '+ str(x1) + ', '+ str(y1) + ', ' + str(width) + ', '+ str(height)+ ', 1, 1, 1')
                    f.write('\n')
            #   #      ----------------------------------------------------------------
            #         kpt = keypoints[j]
            #         cv2.rectangle(img_show, (x1, y1), (x2, y2), color, 1)
            #         cv2.putText(img_show, name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            #         cv2.putText(img_show, str(round(conf_bboxs[j], 2)), (x2, y2-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                count += 1
                print(count)

              except:
                kpts_prev = np.empty((0,51))
                bbox_prev = np.empty((0, 5))
                track_prev = mot_tracker.trackers
                count+=1
                mot_tracker.dead_tracks = []
                continue
        f.close()      
