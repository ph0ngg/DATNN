from sort3 import Sort, KalmanBoxTracker
import cv2
import torch
import numpy as np
import os
import random

from ultralytics import YOLO

torch.cuda.set_device(1)
model = YOLO('yolov8s-pose.pt')
mot_tracker = Sort()
count = 1
frame_size = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#out = cv2.VideoWriter('out_video_4.mp4', (fourcc), 30, frame_size)
kpts_prev = np.empty((0,51))

path = '/home/phongnn/test/test/AIC23'
# for am in ['63am', '64am']:
#     path = os.path.join(folder_path, am)
#     for filename in sorted(os.listdir(path), reverse= False): 
#       if filename[-1] != 'p':
#         print(filename)
#         KalmanBoxTracker.count = 0
#         file_path = os.path.join(path, filename)
#         filesave = filename +'.txt'
#         save_path = os.path.join('/home/phongnn/test/test/sort/result_kpt',am ,str(filesave))
#         f = open(save_path, 'w') 
for j in sorted(os.listdir(path)):
  if j in ['train', 'validation']:
    k = os.path.join(path, j)
    for video in sorted(os.listdir(k)):
      if video in ['S002','S004', 'S006', 'S005']:
        video_path = os.path.join(k, video)
        txt_folder = os.path.join(video_path, 'result_cos_oks')
        for cam in sorted(os.listdir(video_path)):
          if cam[0] == 'c':
            cam_path = os.path.join(video_path, cam)
            for track in sorted(os.listdir(cam_path)):
                if track[-1] == '4':
                    filesave = cam + '.txt'
                    txt_path = os.path.join(txt_folder, str(filesave))
                    track_path = os.path.join(cam_path, track)
                    play = cv2.VideoCapture(track_path)
                    f = open(txt_path, 'w')
                    KalmanBoxTracker.count = 0
                    print(txt_path)
                    while(play.isOpened()):
                            ret, img_show = play.read()
                        # for img in sorted(os.listdir(file_path)):
                        #   if img[-5] == '1':
                        #     img_path = os.path.join(file_path, img)
                        #     img_show = cv2.imread(img_path)
                            #print(img_show.shape)
                            if img_show is None:
                                break
                            preds = model(img_show, verbose = False)
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
                                for tracker in mot_tracker.trackers:
                                  print(tracker.id)
                                num_people = len(track_bbs_ids.tolist()) 
                                coors_later, confs_later, coors_prev, confs_prev = [[]]*num_people, [[]]*num_people, [[]]*num_people, [[]]*num_people
                                for j in range(len(track_bbs_ids.tolist())):
                                    coords = track_bbs_ids.tolist()[j]
                                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                                    #print(coords[4])
                                    name_idx = int(coords[4])-1
                                    name = 'ID: {}'.format(str(name_idx))
                                    color1 = (255, 0, 0)
                                    color2 = (0, 0, 255)
                                    random.seed(int(coords[4]))
                                    color = (255*random.random(), 255*random.random(), 255*random.random())
                                    width = x2 - x1
                                    height = y2 - y1
                                    f.write(str(count) + ', ' +str(name_idx) +', '+ str(x1) + ', '+ str(y1) + ', ' + str(width) + ', '+ str(height)+ ', 1, 1, 1')
                                    f.write('\n')
                              #      ----------------------------------------------------------------
                                    kpt = keypoints[j]
                                    cv2.rectangle(img_show, (x1, y1), (x2, y2), color, 1)
                                    cv2.putText(img_show, name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    cv2.putText(img_show, str(round(conf_bboxs[j], 2)), (x2, y2-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                kpts_prev = keypoints
                                bboxs_prev = bboxs
                                count += 1
                                #print(count)
                                #out.write(img_show)
                              except:
                                kpts_prev = np.empty((0,51))
                                bbox_prev = np.empty((0, 5))
                                count+=1
                                continue
                # f.close()
                    count = 1
                    kpts_prev = np.empty((0,51))
                    bbox_prev = np.empty((0, 5))
                    f.close()

                #out.release()

