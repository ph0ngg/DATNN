from ultralytics import YOLO
import cv2
import os
import json
import numpy as np

folder = '/home/phongnn/test/test/MMPTRACK/train/images'
model = YOLO('yolov8x-pose-p6.pt')
count = 0
for am in sorted(os.listdir(folder)):
    img_folder = os.path.join(folder, am)
    for video in sorted(os.listdir(img_folder)):
        if video == 'industry_safety_0':
            video = os.path.join(img_folder, video)
            for img in sorted(os.listdir(video)):
                if img[-5] == '1':
                  print(img)
                  count+=1
                  if count %30 == 0:
                    img_path = os.path.join(video, img)
                    img_show = cv2.imread(img_path)
                    if img_show is None:
                        break
                    img_show = cv2.resize(img_show, (640, 360))
                    preds = model(img_show, verbose = False)
                    #preds =  preds.astype(np.float64)
                    for r in preds:
                      xy_bboxs = r.boxes.xyxy.cpu().numpy()
                      conf_bboxs = r.boxes.conf.cpu().numpy()
                      xy_keypoints = r.keypoints.xy.cpu().numpy()
                      conf_keypoints = r.keypoints.conf.cpu().numpy()
                      keypoints = np.concatenate((xy_keypoints, np.expand_dims(conf_keypoints, axis = 2)), axis = 2)
                      keypoints = keypoints.reshape(keypoints.shape[0], -1) #shape: num_people, 51
                      zeros = np.zeros((conf_bboxs.shape[0], 1))
                      bboxs = np.concatenate((xy_bboxs, np.expand_dims(conf_bboxs, axis = 1)), axis = 1)
                      res = np.concatenate((bboxs ,zeros, keypoints), axis = 1)
                      dictt = {
                            "data": {
                              "img": '/data/local-files?d=industry_safety_0/' + img
                            },
                            "predictions": [
                              {
                                "result":[]
                              }
                            ]
                      }
                      save_json = os.path.join('/home/phongnn/test/test/sort/keypoint', img[:-4] + '.json')
                      for j in range(len(res)):
                             
                        dictt["predictions"][0]["result"].append(

                                {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 6]/6.4,
                            "y": res[j, 7]/3.6,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "nose"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                                {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 21]/6.40,
                            "y": res[j, 22]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "left_shoulder"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                              {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 24]/6.40,
                            "y": res[j, 25]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "right_shoulder"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                            {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 27]/6.40,
                            "y": res[j, 28]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "left_elbow"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                            {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 30]/6.40,
                            "y": res[j, 31]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "right_elbow"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                                                        {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 33]/6.40,
                            "y": res[j, 34]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "left_wrist"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                                                        {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 36]/6.40,
                            "y": res[j, 37]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "right_wrist"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                                                        {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 39]/6.40,
                            "y": res[j, 40]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "left_hip"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                                                        {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 42]/6.40,
                            "y": res[j, 43]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "right_hip"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(                         
                         {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 45]/6.40,
                            "y": res[j, 46]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "left_knee"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                                                        {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 48]/6.40,
                            "y": res[j, 49]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "right_knee"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                          {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 51]/6.40,
                            "y": res[j, 52]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "left_ankle"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                                                  {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 54]/6.40,
                            "y": res[j, 55]/3.60,
                            "width": 0.22471910112359553,
                            "keypointlabels": [
                              "right_ankle"
                            ]
                          },
                          "from_name": "kp-1",
                          "to_name": "img-1",
                          "type": "keypointlabels",
                          "origin": "manual"
                        })
                        dictt["predictions"][0]["result"].append(
                                                  {
                          "original_width": 640,
                          "original_height": 360,
                          "image_rotation": 0,
                          "value": {
                            "x": res[j, 0]/6.40,
                            "y": res[j, 1]/3.60,
                            "width": (res[j, 2]-res[j, 0])/6.4,
                            "height": (res[j, 3] - res[j, 1])/3.6,
                            "rectanglelabels": [
                              "People"
                            ]
                          },
                          "from_name": "label",
                          "to_name": "img-1",
                          "type": "rectanglelabels",
                          "origin": "manual"
                        })
                              
                      with open(save_json, 'w') as f:
                        json_string = json.dumps(dictt, indent = 4)
                        f.write(json_string)
                        f.write('\n')
                        f.close

