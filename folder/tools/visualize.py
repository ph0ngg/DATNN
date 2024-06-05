import cv2
import os
import random


def generate_color():
    # Hàm tạo màu ngẫu nhiên theo định dạng BGR
    blue = random.randint(0, 255)
    green = random.randint(0, 255)
    red = random.randint(0, 255)
    return (blue, green, red)

n = 100
color_dict = color_dict = {i: generate_color() for i in range(1, n + 1)}


# count_frame = 0
# for cam in sorted(os.listdir('/home/phongnn/test/test/AIC23/train/S002')):
#   if cam[0] == 'c':
#     video_path = f'/home/phongnn/test/test/AIC23/train/S002/{cam}/video.mp4'
    
#     f = open(f'/home/phongnn/test/test/AIC23/train/S002/result_reid/{cam}_new.txt', 'r')
#     my_dictt = {}
#     lines = f.readlines()
#     for line in lines:
#         x = line.split(',')
#         if x[0] not in my_dictt:
#           my_dictt[x[0]] = [(x[1], x[2], x[3], x[4], x[5])]
#         else:
#           my_dictt[x[0]].append((x[1], x[2], x[3], x[4], x[5]))
#     video = cv2.VideoCapture(video_path)
#     save_path = f'/home/phongnn/test/test/AIC23/train/S002/result_reid/{cam}.mp4'

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     frame_size = (1920, 1080)
#     out_video = cv2.VideoWriter(save_path, (fourcc), 30, frame_size)
#     while(video.isOpened()):
#         ret, img_show = video.read()
#         print(count_frame)
#         if img_show is None:
#             break
#         try:
#           for x in my_dictt[str(count_frame)]:
#             id, x1, y1, width, height = x
#             x1 = int(x1)
#             y1 = int(y1)
#             width = int(width)
#             height = int(height)
#             x2 = x1 + width
#             y2 = y1 + height
#             #color = (255*random.random(), 255*random.random(), 255*random.random())
#             color = (255, 0, 0)
#             name = 'ID: {}'.format(str(id))
#             cv2.rectangle(img_show, (x1, y1), (x2, y2), color, 1)
#             cv2.putText(img_show, name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#           count_frame += 1
#           out_video.write(img_show)
#         except:
#           count_frame += 1
#           out_video.write(img_show)
#           continue
        
        
#     out_video.release()


#-------------
count_frame = 0
for cam in sorted(os.listdir('/home/phongnn/test/test/Campus4')):
  if cam[-1] == 'i':
    video_path = os.path.join('/home/phongnn/test/test/Campus4', cam)
    
    f = open(f'/home/phongnn/test/test/Campus4/result_reid/{cam[:-4]}_new.txt', 'r')
    my_dictt = {}
    lines = f.readlines()
    for line in lines:
        x = line.split(',')
        if x[0] not in my_dictt:
          my_dictt[x[0]] = [(x[1], x[2], x[3], x[4], x[5])]
        else:
          my_dictt[x[0]].append((x[1], x[2], x[3], x[4], x[5]))
    video = cv2.VideoCapture(video_path)
    save_path = f'/home/phongnn/test/test/Campus4/out_video/{cam[:-4]}.mp4'

    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    vid_width, vid_height = int(video.get(3)), int(video.get(4))
    frame_size = (vid_width, vid_height)
    fps = video.get(cv2.CAP_PROP_FPS)
    out_video = cv2.VideoWriter(save_path, (cv2.VideoWriter_fourcc(*'mp4v')), fps, frame_size)
    print(save_path)
    while(video.isOpened()):
        ret, img_show = video.read()
        if img_show is None:
            break
        try:
          for x in my_dictt[str(count_frame)]:
            id, x1, y1, width, height = x
            x1 = int(x1)
            y1 = int(y1)
            width = int(width)
            height = int(height)
            x2 = x1 + width
            y2 = y1 + height
            #color = (255*random.random(), 255*random.random(), 255*random.random())
            color = color_dict[int(id)+1]
            #color = (255, 0, 0)
            name = 'ID: {}'.format(str(id))
            cv2.rectangle(img_show, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_show, name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
          count_frame += 1
          out_video.write(img_show)
          #print(count_frame)
        except:
          count_frame += 1
          out_video.write(img_show)
          continue
        
    out_video.release()

