# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset
from PIL import Image, ImageOps


from .data_utils import read_image
import numpy as np
import cv2
import torch
import json

class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
        
        with open('/home/phongnn/test/test/market1501/Market-1501-v15.09.15/label1.json', 'r') as f:
            self.view_label = json.load(f)

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)
#         # -------------------------
# # #/home/phongnn/test/test/market1501/Market-1501-v15.09.15/bounding_box_train/0002_c1s1_000451_03.jpg
#         file_name = img_path.split('/')[-1]
#         heatmaps = ['/home/phongnn/test/test/market1501/heatmap/' + file_name[:-4] + '_heatmap_' + str(i) + '.jpg' for i in range(1, 18) if i not in [1, 2, 3, 4]]
#         heatmap_grays = []
#         heatmap_total = np.zeros((128, 64))
#         for heatmap in heatmaps:

#             heatmap = cv2.imread(heatmap)
#             heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
#             #heatmap_gray = heatmap_gray[:, :, np.newaxis]
#             heatmap_grays.append(heatmap_gray)
#         heatmap_grays = np.stack(heatmap_grays, axis = -1)
#         #img = np.array(img)
#         img = np.concatenate((img, heatmap_grays), axis = -1)
#         img = np.transpose(img,(2, 0, 1))
# #        img = img.float()
#         img = torch.from_numpy(img)
#         # if self.transform is not None:
#         #     img = self.transform(img)
#             #img = ColorJitter(img)

#         #print(img.shape)
#         #img = torch.from_numpy(img)
#         # -------------------------
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
           # "view_label": self.view_label[img_path]
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
