from pymongo import MongoClient
import numpy as np
import urllib.parse
from matplotlib.gridspec import GridSpec
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import numpy as np

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
    
def _plot_kmean_scatter(X, labels, gs, thres):
    '''
    X: dữ liệu đầu vào
    labels: nhãn dự báo
    '''
    # lựa chọn màu sắc
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("hls", num_classes))

    # vẽ biểu đồ scatter
    ax = plt.subplot(gs)
    sc = ax.scatter(X[:,0], X[:,1], lw=0, s=40, c=palette[labels.astype(np.int32)])

    # thêm nhãn cho mỗi cluster
    txts = []

    for i in range(num_classes):
        # Vẽ text tên cụm tại trung vị của mỗi cụm
        indices = (labels == i)
        xtext, ytext = np.median(X[indices, :], axis=0)
        if not (np.isnan(xtext) or np.isnan(ytext)):        
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    plt.title('t-sne visualization for thres={:.4f}'.format(thres))
    plt.savefig('image.png')

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
emb_threshold = 0.2


vector_embeddings = db['S007_emb_vector']
global_id = db['S007_gid']
cluster_centers = db['cluster_centers']
db = []
my_set = set()
result = '/home/phongnn/test/test/fast-reid/tools/reid.txt'
#f = open(result, 'w')
print('Reading DB.....')
for i, doc1 in enumerate(vector_embeddings.find()):
    #print(doc1['key'])
    key1 = doc1['key']
    value1 = doc1['embedding_vector']
    # if key1 == 700:
    #     break
    for m in value1:
        db.append(m[:-1])

print('Done Reading DB, start clustering')
print(np.array(db).shape)
#infer(np.array(db))
for i, samples in enumerate(np.linspace(50, 200, 4)):
    print(f'samples: {samples}')
    dbscan = DBSCAN(eps=0.142, min_samples=50, metric='cosine')
    labels = dbscan.fit_predict(db)
    print(np.unique(labels))

# dbscan = DBSCAN(eps=0.142, min_samples=50, metric='cosine')
# labels = dbscan.fit_predict(db)
# for i in range(len(labels)):
#     labels[i] = int(labels[i]) 
# for label in np.unique(labels):
#     if label == -1:
#         continue
#     cluster_points = np.array(db)[labels == label]
#     cluster_center = np.mean(cluster_points, axis=0)
#     for m in range(len(cluster_center)):
#         cluster_center[m] = float(cluster_center[m])
#     cluster_center = list(cluster_center)
#     my_dictt = {"label": int(label), "center": cluster_center}
#     cluster_centers.insert_one(my_dictt)
