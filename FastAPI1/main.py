from fastapi import FastAPI, Request, File, UploadFile, WebSocket, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import aiofiles
from FastAPI1.engine import tracking, reid, get_reid_txt_file, visualize
sys.path.append('./sort')
from sort import Sort, KalmanBoxTracker
import random
import base64

app = FastAPI()

origins = ["http://localhost:3000",
           "http://localhost:8000/upload/",
           "http://localhost:8000/process_video/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.get("/video")
# async def get_video():
#     video_path = "d:\Phong\cam3.mp4"
#     return FileResponse(video_path, media_type="video/mp4")


# Đường dẫn đến thư mục lưu trữ tệp tải lên
UPLOAD_FOLDER = "./upload_folder"
SAVED_FOLDER = "./result/result_video"

@app.post("/upload/")
async def upload_videos(videos: List[UploadFile] = File(...)):
    upload_videos_paths = []
    processed_video_paths = []
    for video in videos:
        file_path = os.path.join(UPLOAD_FOLDER, video.filename)
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(await video.read())
        upload_videos_paths.append(file_path)
        saved_path = os.path.join(SAVED_FOLDER, video.filename)
        processed_video_paths.append(saved_path)
    process_video(upload_videos_paths)  # Hàm process_video() là hàm xử lý video 
    return {"processed_video_paths": processed_video_paths}

def process_video(folder):
    #Xử lý video ở đây và trả về đường dẫn của video đã được xử lý
    tracking.tracking(folder)
    reid.reid()
    get_reid_txt_file.get_reid_txt_file()
    visualize.visualize()
    #tracking --> reid --> get_reid_text_file --> visualize
    
@app.get("/processed_video/{filename}")
async def get_processed_video(filename: str):
    processed_video_path = os.path.join('./result/result_video', filename[:-4] + '.mp4')
    #processed_video_path = "./result/result_video/4p-c1.avi"
    return FileResponse(processed_video_path)

UPLOAD_FOLDER_SCT = './upload_folder_sct'
@app.post("/uploadsct/")
async def upload_videos(videos: List[UploadFile] = File(...)):
    upload_videos_paths = []
    global cap
    for video in videos:
        file_path = os.path.join(UPLOAD_FOLDER_SCT, video.filename)
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(await video.read())
        cap = cv2.VideoCapture(file_path)
    return {"processed_video_paths": upload_videos_paths}

mot_tracker = Sort()
detect_kpt_model = YOLO('yolov8s-pose.pt')
@app.get("/video_stream/")
def stream_frame():
    global cap
    #file_path = os.path.join(UPLOAD_FOLDER_SCT, os.listdir(UPLOAD_FOLDER_SCT)) 
    if cap is None or not cap.isOpened():
        return Response(status_code= 204)
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            return Response(status_code=204)
            cap.release()
        preds = detect_kpt_model(frame, verbose = False)
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
                for j in range(len(track_bbs_ids.tolist())):
                    coords = track_bbs_ids.tolist()[j]
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    name_idx = int(coords[4])-1
                    name = 'ID: {}'.format(str(name_idx))
                    color1 = (255, 0, 0)
                    color2 = (0, 0, 255)
                    random.seed(int(coords[4]))
                    color = (255*random.random(), 255*random.random(), 255*random.random())
                    width = x2 - x1
                    height = y2 - y1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(frame, name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                return Response(content=frame_bytes, media_type='image/jpeg')

            except:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                return Response(content=frame_bytes, media_type='image/jpeg')

# SCT tất cả video trước, tạo n embedding cluster trong db, gộp vào rồi reid
