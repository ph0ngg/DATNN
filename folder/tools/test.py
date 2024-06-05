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
def person_embedding(model, imgs):
    # img = img/255
    # img[:, :, 0] = img[:, :, 0] - 0.485
    # img[:, :, 1] = img[:, :, 1] - 0.456
    # img[:, :, 2] = img[:, :, 2] - 0.406
    # img[:, :, 0] /= 0.229
    # img[:, :, 1] /= 0.224
    # img[:, :, 2] /= 0.225
    #B, H, W, C
    imgs = np.transpose(imgs, (0, 3, 1, 2)) #(B, C, H, W)
    imgs = torch.from_numpy(imgs)
    imgs = imgs.type(torch.FloatTensor)  
    imgs = imgs.cuda()
    # img = cv2.resize(img, (64, 128))
    # img = np.transpose(img, (2, 0, 1))
    # img = np.expand_dims(img, axis= 0)
    # img = torch.from_numpy(img)
    # img = img.type(torch.FloatTensor).cuda()
    
    # feature = model.backbone(img)[0]
    # person_emb = model.heads(feature).detach().cpu().numpy()
    # person_emb = model(imgs).detach().cpu().numpy()
    person_emb = model(imgs)
    person_emb = F.normalize(person_emb)
    person_emb = person_emb.cpu().data.numpy()
    view_prop = model.backbone(imgs)[1].detach().cpu().numpy()
    view = np.argmax(view_prop)
    return person_emb, view

def crop_img(img, x1, y1, x2, y2):
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    cropped_img = img[int(y1):int(y2), int(x1):int(x2)] #H, W
    cropped_img = cv2.resize(cropped_img, (64, 128))

    return cropped_img

def distance(vt1, vt2, dist_type = 'cosine'):
    dist = 0
    if dist_type == 'euclid':
        for i in range(len(vt1)):
            dist += (vt1[i]-vt2[i])**2
        return sqrt(dist)
    elif dist_type == 'cosine':
        dist = np.dot(vt1,vt2)/(np.linalg.norm(vt1)*np.linalg.norm(vt2))
        return 1 - dist

def min_max_matching_distance(model):
    path = '/home/phongnn/test/test/AIC23/train/S002/query'
    imgs = os.listdir(path)
    max_dist = 0
    for i in range(len(imgs)):
        print(i)
        if imgs[i].split('_')[0] == '9':
            img1 = cv2.imread(os.path.join(path, imgs[i]))
            emb1 = person_embedding(model, img1)[0]
            for j in range(i, len(imgs)):
                if imgs[j].split('_')[0] == '9':
                    img2 = cv2.imread(os.path.join(path, imgs[j]))
                    emb2 = person_embedding(model, img2)[0]
                    dist = distance(emb1, emb2)
                    if max_dist < dist:
                        max_dist = dist
    print(max_dist)

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



def test_distance(img1, img2, model):
    img1 = np.expand_dims(img1, 0)
    img2 = np.expand_dims(img2, 0)
    emb1 = person_embedding(model, img1)[0][0]
    emb2 = person_embedding(model, img2)[0][0]
    dist = distance(emb1, emb2, 'cosine')
    return dist

def score(bbox, kpt):
    conf = 0
    for i in range(2, len(kpt), 3):
        conf += kpt[i]
    bbox_size_score = bbox.shape[0] * bbox.shape[1] / (128*64)
    bbox_ratio_score = (bbox.shape[1]/bbox.shape[0]) / 2
    return bbox_ratio_score*bbox_size_score + conf

def infer(model, df):
    countt = 0
    embs = []  
    #df = pd.DataFrame(columns= ['Embeddings', 'Label'])
    
    # for path in (sorted(os.listdir('/home/phongnn/test/test/AIC23/train/S002/query'))):
    #     if path[0] != 'T':
    #         id_img = (path.split('_')[0])
    #         if int(id_img) <= 100:
    #             img_path = os.path.join('/home/phongnn/test/test/AIC23/train/S002/query', path)
    #             img = cv2.imread(img_path)
    #             try:
    #                 vt_emb = person_embbeding(model, img)[0]
    #                 embs.append(vt_emb)
    #                 df.loc[len(df.index)] = [vt_emb, id_img]
    #                 countt+=1
    #                 print(countt)
    #             except:
    #                 continue
    X = np.array(df['Embeddings'].to_list(), dtype=np.float32)
    print(X.shape)
    tsne = TSNE(random_state=0, n_iter=1000)
    tsne_results = tsne.fit_transform(X)
    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df_tsne['Class Name'] = df['Label']
    fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
    sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
    custom_palette = sns.color_palette("hsv", 50)
    sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Class Name', palette=custom_palette)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title('Scatter plot of news using t-SNE')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.axis('equal')
    plt.savefig('img.png')


def main(args):
    cfg = setup(args)
    emb_model = build_model(cfg)
    emb_threshold = 0.35
    print(1)
    MONGO_HOST = "192.168.1.220"
    MONGO_PORT = 27017
    MONGO_DB = "mydb"
    MONGO_USER = "awlvn"
    MONGO_PASS = urllib.parse.quote_plus("123456a@")

    #Creating a pymongo client
    url = "mongodb://{}:{}@{}:{}/{}?authSource=admin".format(MONGO_USER, MONGO_PASS, MONGO_HOST, MONGO_PORT, MONGO_DB)
    client = MongoClient(url)
    db = client[MONGO_DB]
    emb_db = db['Campus4']
    state_dict = torch.load('/home/phongnn/test/test/fast-reid/tools/model_loss/khongconv5_lossview0,5.pth')

    emb_model.load_state_dict(state_dict['model'])
    emb_model.eval()
    detect_kpt_model = YOLO('yolov8s-pose.pt')
    mot_tracker = Sort()
    track_prev = []

    emb = {}
    start_times = {}
    count = 0
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (1920, 1080)

    # img1 = cv2.imread('/home/phongnn/test/test/EPFL/result_reid/track_img/438_6.jpg')
    # img2 = cv2.imread('/home/phongnn/test/test/EPFL/result_reid/track_img/430_8.jpg')

    # img5 = cv2.imread('/home/phongnn/test/test/EPFL/result_reid/track_img/8_5.jpg')
    # print('Same person, distance = ',test_distance(img5, img1, emb_model))
    #print(person_embbeding(emb_model, img1)[1])
    #min_max_matching_distance(emb_model)
#     X = pd.DataFrame(columns = ['Embeddings', 'Label'])
#     for track in emb_db.find():
#         new_data = {'Embeddings': track['embedding_vector'], 'Label': track['key']}
#         if track['key'] < 50:
#             X = pd.concat([X, pd.DataFrame(new_data)], ignore_index=True)
#     #infer(emb_model, X)

#----------------
    for cam in sorted(os.listdir('/home/phongnn/test/test/Campus4')):
      #if cam in ['4p-c2.avi', '4p-c3.avi', '4p-c0.avi', '4p-c1.avi']:
        cam_path = os.path.join('/home/phongnn/test/test/Campus4', cam)
        txt_folder = os.path.join('/home/phongnn/test/test/Campus4', 'result_reid')
        #video_path = os.path.join(cam_path, 'video.mp4')
        video_path = cam_path
        video_save = os.path.join(txt_folder, cam)# + '.mp4')
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        vid_width, vid_height = int(video.get(3)), int(video.get(4))
        frame_size = (vid_width, vid_height)
        filesave = cam[:-4] + '.txt'
        txt_path = os.path.join(txt_folder, str(filesave))
        f = open(txt_path, 'w')
        print(txt_path)
        
        id_appear = set()
        time1= time.time()
        while video.isOpened():
            ret, img_show = video.read()
            if img_show is None:
                for track in track_prev:
                    try:
                        save_img = emb[track.id][:10] #chọn 10 ảnh có kích cỡ to nhất/
                        emb_doc = []
                        bboxs, kpts = zip(*[(x[0], x[1]) for x in save_img])
                        views = np.array(view_predict(kpts))
                        views_reshape = np.reshape(views, (-1, 1))
                        emb_vectors = list(person_embedding(emb_model, bboxs)[0])
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
# ---------------------------------------------------------------
             #   mot_tracker: Danh sách các track trong frame hiện tại
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
                        emb_vectors = list(person_embedding(emb_model, bboxs)[0])
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
                
                #out.write(img_show)
              except:
                kpts_prev = np.empty((0,51))
                bbox_prev = np.empty((0, 5))
                track_prev = mot_tracker.trackers
                count+=1
                mot_tracker.dead_tracks = []
               # out.write(img_show)
                continue
        time2 = time.time()
        print(time2-time1)
        #out.release()
#     #person_emb= person_embbeding(img)

        
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
