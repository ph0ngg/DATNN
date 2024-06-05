import pymongo
from pymongo import MongoClient
import numpy as np
import urllib.parse
from matplotlib.gridspec import GridSpec
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cosine, cdist

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


def distance(vt1, vt2, dist_type = 'cosine'):
    dist = 0
    if dist_type == 'euclid':
        for i in range(len(vt1)):
            dist += (vt1[i]-vt2[i])**2
        return np.sqrt(dist)
    elif dist_type == 'cosine':
        dist = np.dot(vt1,vt2)/(np.linalg.norm(vt1)*np.linalg.norm(vt2))
        return 1 - dist
    
def infer(db):
    countt = 0
    embs = []  
    df = pd.DataFrame(columns= ['Embeddings', 'Label'])
    for m in db:
        df.loc[len(df.index)] = [list(m.values())[0], list(m.keys())[0]]
        countt +=1
        print(countt)
    #print(df)
    # for path in (sorted(os.listdir('/home/phongnn/test/test/market1501/Market-1501-v15.09.15/bounding_box_train'))):
    #     if path[0] != 'T':
    #         id_img = (path.split('_')[0])
    #         if int(id_img) <= 100:
    #             img_path = os.path.join('/home/phongnn/test/test/market1501/Market-1501-v15.09.15/bounding_box_train', path)
    #             img = cv2.imread(img_path)
    #             vt_emb = person_embbeding(model, img)[0]
    #             embs.append(vt_emb)
    #             df.loc[len(df.index)] = [vt_emb, id_img]
    #             countt+=1
    #             print(countt)
    X = np.array(df['Embeddings'].to_list(), dtype=np.float32)
    #print(X.shape)
    tsne = TSNE(random_state=0, n_iter=1000)
    tsne_results = tsne.fit_transform(X)
    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df_tsne['Class Name'] = df['Label']
    fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
    sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
    sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Class Name', palette='hls')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title('Scatter plot of news using t-SNE')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.axis('equal')
    plt.savefig('img2.png')


MONGO_HOST = "192.168.1.220"
MONGO_PORT = 27017
MONGO_DB = "mydb"
MONGO_USER = "awlvn"
MONGO_PASS = urllib.parse.quote_plus("123456a@")
#Creating a pymongo client
url = "mongodb://{}:{}@{}:{}/{}?authSource=admin".format(MONGO_USER, MONGO_PASS, MONGO_HOST, MONGO_PORT, MONGO_DB)
client = MongoClient(url)
db = client[MONGO_DB]
emb_threshold = 0.3



vector_embeddings_db = db['Campus4']
global_id = db['Campus4_gid']
db = []
my_set = set()

gallery = {}
current_id = 1
dictt2 = {}
vector_embeddings = vector_embeddings_db.find({}).sort('key', pymongo.ASCENDING)
first_person = vector_embeddings[0]
gallery[0] = [first_person["embedding_vector"], first_person['start_time'], first_person['end_time']] # gallery lưu sẵn id 0 có vector embedding của track đầu tiên
for i, track in enumerate(vector_embeddings):    #duyệt từng track trong db
    print(i)
    #vector_embeddings = [row[:-1] for row in track['embedding_vector']] 
    vector_embedding = track['embedding_vector'] #emb vector của từng track
    start_time = track['start_time']
    end_time = track['end_time']
    vectors1 = [row[:-1] for row in vector_embedding]
    min_dist = 1
    this_gid = 0
    same_view = True
    this_same_view = True

    for gid in sorted(gallery):  #duyệt global id trong gallery
        #print('gallery[gid].shape: ', np.array(gallery[gid]).shape)
        if start_time <= gallery[gid][2]:  #start time cua track hien tai so sanh voi end time cua gid trong qua khu
            continue
        gid_emb_vectors = gallery[gid][0]
        vectors2 = [row[:-1] for row in gid_emb_vectors]
        #gid_emb_vectors = gallery[gid] #lấy ra embedding vectors của global id đang duyệt
        cosine_matrix = cdist(vectors1, vectors2, 'cosine')
        dist = np.min(cosine_matrix)
        index = np.unravel_index(np.argmin(cosine_matrix), cosine_matrix.shape)
        same_view = (vector_embedding[index[0]][-1] == gid_emb_vectors[index[1]][-1])
        if dist < min_dist:
            this_gid = gid #gid mà có khoảng cách với track hiện tại là min
            min_dist = dist
            this_same_view = same_view

    if this_same_view == True:
        emb_threshold = 0.4
    else:
        emb_threshold = 0.52
    if min_dist < emb_threshold:
        gallery[this_gid][1] = start_time
        gallery[this_gid][2] = end_time
        if len(gallery[this_gid][0]) < 250:
            gallery[this_gid][0].extend(vector_embedding)

    else: #min dist > threshold, track la track moi
        print(min_dist)
        gallery[current_id] = [vector_embedding, start_time, end_time]
        this_gid = current_id
        current_id += 1
    my_dictt = {"old key": track['key'], "new_key": this_gid}
    dictt2[this_gid] = (track['key'], start_time, end_time)  #mảng lưu trữ các track id của một global id
    global_id.insert_one(my_dictt)