import numpy as np
import cv2
import os

path = '/home/phongnn/test/test/sort/data/data'
gts_path = os.path.join(path, 'gt')
videos_path = os.path.join(path, 'video')

for filename in os.listdir(videos_path):
    print(filename)
    video_path = os.path.join(videos_path, filename)
    gt_path = os.path.join(gts_path, filename)[:-4] + '.txt'
    f = open(gt_path)
    gt = f.readlines()
    for i in range(len(gt)):
        gt[i] = gt[i].split(',')
    dictt = {}
    for sublist in gt:
        key = sublist[0]  
        value = [sublist[1], sublist[2], sublist[3], sublist[4], sublist[5]]  

        if key in dictt:
            dictt[key].append(value)
        else:
            dictt[key] = [value]      
    video = cv2.VideoCapture(video_path)
    #current_frame = gt[0][0]
    cnt = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cnt += 1
        if (cnt %5 ==0):
          try:
              for i in dictt[str(cnt)]:
                  id = i[0]
                  x, y, w, h = float(i[1]), float(i[2]), float(i[3]), float(i[4])
                  xmin = int(x)
                  xmax = int(x + w)
                  ymin = int(y)
                  ymax = int(y + h)
                  img = frame[ymin:ymax, xmin:xmax, :]
                  cid = 'c' + filename[1]
                  sid = 's' + filename[7]
                  cv2.imwrite(os.path.join('/home/phongnn/test/test/sort/data/data/img', id+ '_' + cid + sid +'_'+ str(cnt) + '_00') + '.jpg', img)
          except:
              pass