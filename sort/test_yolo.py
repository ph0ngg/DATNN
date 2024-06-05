from ultralytics import YOLO
import cv2
import numpy as np

#model = YOLO('yolov8s-pose.pt', verbose = False)
img_path = '/home/phongnn/test/test/sort/out.jpg'
img = cv2.imread(img_path)
print(img.shape)
# #img = ((img/255)**(0.5)*255).astype(np.uint8)
# pred = model(img)
# for result in pred:
#     keypoints = result.keypoints.xy.cpu().numpy()  # Keypoints object for pose outputs
#     print(keypoints.shape)
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
