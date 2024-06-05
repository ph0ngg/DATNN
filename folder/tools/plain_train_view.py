import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os, sys
import torch.nn.functional as F
#from yolo_pose_onnx import *
sys.path.append('/home/phongnn/test/test/fast-reid')
from tqdm import tqdm
from PIL import Image
from tempfile import TemporaryDirectory
from fastreid.modeling.backbones import build_osnet_backbone, build_shufflenetv2_backbone
import cv2

from ultralytics import YOLO
model = YOLO('yolov8s-pose.pt')


def heat_map(image, folder):
  #đầu vào image là (channel, width, height)
    img = os.path.join('/home/phongnn/test/test/market1501/Market-1501-v15.09.15', folder, image + '.jpg')
    img = cv2.imread(img)
    #pred = read_model(img)
    pred = model(img, verbose = False)
    for r in pred:
        xy_keypoints = r.keypoints.xy.cpu().numpy()

    img_width = 64
    img_height = 128
    keypoints = []
    heatmaps = []
    point_size = 10
    gaussian_radius = 5

    if xy_keypoints.shape[0] == 1:
        # for rows in pred:
        #     if rows[4] == max(pred[:, 4]):
        #         new_preds.append(rows)
        # new_preds = np.array(new_preds)
        #print(new_preds)
        # for i in range(6, 57, 3):
        #         new_preds[:, i] = new_preds[:, i] / 640.0 * img_width
        #         new_preds[:, i+1] = new_preds[:, i+1] / 640.0 * img_height
        #        keypoints.append((new_preds[:, i], new_preds[:, i+1]))
                #print(keypoints[0], new_preds[:, 4])
        for keypoint in xy_keypoints.reshape(17, 2):
            x, y = keypoint
            #print(x, y)
            heatmap = np.zeros((img_height, img_width))
            for i in range(len(x)):
               # try:
                gaussian_kernel = cv2.getGaussianKernel(gaussian_radius * 2 + 1, 0)
                gaussian_kernel = gaussian_kernel * gaussian_kernel.T
        # Đặt Gaussian kernel vào ma trận heatmap ở vị trí (x, y)
                try:
                    heatmap[int(y[i]) - gaussian_radius:int(y[i]) + gaussian_radius + 1, int(x[i]) - gaussian_radius:int(x[i]) + gaussian_radius + 1] = gaussian_kernel * 25
                except:
                    pass    
                heatmaps.append(heatmap)
               # except:
            heatmaps_array = np.array(heatmaps)
            for i, heatmap in enumerate(heatmaps_array):
                heatmap = heatmap[:, :, np.newaxis]
                name = '/home/phongnn/test/test/market1501/heatmap'
                cv2.imwrite(os.path.join(name ,f'{image}_heatmap_{i+1}.jpg'), heatmap * 255)
    else:        
        heatmap = np.zeros((img_height, img_width))
        heatmaps = [heatmap]*17
        name = '/home/phongnn/test/test/market1501/heatmap' 
        for i in range(17):
            cv2.imwrite(os.path.join(name ,f'{image}_heatmap_{i+1}.jpg'), heatmaps[i] * 255)
#   new_img = np.stack(new_img, axis = -1)
#   print(new_img.shape)
#   return torch.from_numpy(np.array(new_img))

            # cv2.imwrite(os.path.join('/mnt/hdd3tb/Users/phongnn/test/Market1203/Market1203_orient_3classes_split/heatmap' ,f'heatmap_{i+1}.png'), heatmaps * 255)
for folder in os.listdir('/home/phongnn/test/test/market1501/Market-1501-v15.09.15'):
    print(folder)
    if (folder in ['gt_bbox', 'bounding_box_test', 'bounding_box_train', 'query']):
        for i, img in tqdm(enumerate(sorted(os.listdir(os.path.join('/home/phongnn/test/test/market1501/Market-1501-v15.09.15', folder))))):
    #if i < 100:
    
    #if img < '0657_c1s3_053851_00.jpg' and img > '0555_c1s3_011946_00.jpg':
        # print(img)
          if img != 'Thumbs.db':
            img = img[:-4]
            heat_map(img, folder)

# class horizontal(torch.nn.Module):
#     def __init__(self, p = 0.5):
#         self.p = p
#     def forward(self, img):
#         image = img[:, :, :3]
#         if torch.rand(1) < self.p:
#             return F.hflip(image)
#         img[:, :, :3] = image
#         return img

# class vertical(torch.nn.Module):
#     def __init__(self, p = 0.5):
#         self.p = p
#     def forward(self, img):
#         image = img[:, :, :3]
#         if torch.rand(1) < self.p:
#             return F.vflip(image)
#         img[:, :, :3] = image
#         return img 

# data_transforms ={
#     'train': transforms.Compose([
#         #transforms.RandomResizedCrop(224),
#         #vertical(0.5),t
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomErasing(p = 0.5),
#         transforms.RandomAffine(10),

#         #transforms.ColorJitter(brightness = 0.5, hue = .3),
#         #transforms.RandomPosterize(bits= 2)
#         #transforms.ToTensor(),
#         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#     #     transforms.RandomPosterize(bits= 2)

#     #     #transforms.Resize(256),
#     #     #transforms.CenterCrop(224),
#     #     #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }




# # data_dir = '/mnt/hdd3tb/Users/phongnn/test/Market1203/Market1203_orient_3classes_split'
# # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
# #                                           data_transforms[x],
# #                                           )
# #                   for x in ['train', 'val']}
# def make_dataset(directory):
#     class_to_idx = {'0_front':0,
#                     '1_side':1,
#                     '2_back':2}
#     instances = []
#     available_classes = set()
#     for target_class in sorted(class_to_idx.keys()):
#         class_index = class_to_idx[target_class]
#         target_dir = os.path.join(directory, target_class)
#         if not os.path.isdir(target_dir):
#             continue
#         for root, _, fnames in sorted(os.walk(target_dir, followlinks= True)):
#             for fname in sorted(fnames):
#                 path = os.path.join(root, fname)
#                 item = path, class_index
#                 instances.append(item)

#                 if target_class not in available_classes:
#                     available_classes.add(target_class)
    
#     return instances

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, root, transform):
#         self.classes = ['0_front', '1_side', '2_back']
#         self.class_to_idx = {'0_front':0,
#                             '1_side':1,
#                             '2_back':2}
#         self.root = root
#         self.transform = transform
#         self.samples = make_dataset(root)
#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         file_name = path.split('/')[-1]
#         # heatmaps = ['/home/phongnn/test/test/Market1203/Market1203_orient_3classes_split/heatmap/' + file_name + '_heatmap_' + str(i) + '.jpg' for i in range(1, 18) if i not in [1, 2, 3, 4]]
#         # #heatmap co shape la (128, 64, 3) - (height-width-channel)
#         # #chuyen ve anh xam
#         # heatmap_grays = []
#         # for heatmap in heatmaps:
#         #     heatmap = cv2.imread(heatmap)
#         #     heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
#         #     #heatmap_gray = heatmap_gray[:, :, np.newaxis]
#         #     heatmap_grays.append(heatmap_gray)
#         # heatmap_grays = np.stack(heatmap_grays, axis = -1)
#         img = cv2.imread(path)
#         #img = Image.fromarray(img)
#         #img = np.transpose(img,(2, 0, 1))

#         # #print('heatmap_grays shape', np.array(heatmap_grays).shape)
#         #img = np.concatenate((img, heatmap_grays), axis = -1)
#         img = np.transpose(img,(2, 0, 1))
#         # #img = self.transform(img)
#         img = torch.from_numpy(img)
#         img = self.transform(img)
#         target = torch.tensor(target)
#         #target = self.transform(target)
#         return img, target
    
#     def __len__(self):
#         return len(self.samples)
    
# data_dir = '/home/phongnn/test/test/Market1203/Market1203_orient_3classes_split'    
# image_datasets = {x: CustomDataset(os.path.join(data_dir, x), transform= data_transforms[x])
#                   for x in ['train', 'val']} 
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
#                                              shuffle=True, num_workers=2)
#                for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#         since = time.time()

#     # Create a temporary directory to save training checkpoints
#     # save_dir = '/mnt/hdd3tb/Users/phongnn/test/fast-reid'
#     # with TemporaryDirectory() as tempdir:
#     #     best_model_params_path = os.path.join(save_dir, 'best_model_params.pt')

#     #     torch.save(model.state_dict(), best_model_params_path)
#         best_acc = 0.0

#         for epoch in range(num_epochs):
#             print(f'Epoch {epoch}/{num_epochs - 1}')
#             print('-' * 10)

#             # Each epoch has a training and validation phase
#             for phase in ['train', 'val']:
#                 if phase == 'train':
#                     model.train()  # Set model to training mode
#                 else:
#                     model.eval()   # Set model to evaluate mode

#                 running_loss = 0.0
#                 running_corrects = 0

#                 # Iterate over data.
#                 for inputs, labels in tqdm(dataloaders[phase]):
#                     #print(inputs.shape)
# #                    inputs = heat_map(inputs)
#                     #print(inputs.shape)
#                     inputs = inputs.to(device).float()
#                     labels = labels.to(device)

#                     # zero the parameter gradients
#                     optimizer.zero_grad()
#                     # forward
#                     # track history if only in train
#                     with torch.set_grad_enabled(phase == 'train'):
#                         outputs = model(inputs)[1]
#                         _, preds = torch.max(outputs, 1)
#                         loss = criterion(outputs, labels)
#                         # backward + optimize only if in training phase
#                         if phase == 'train':
#                             loss.backward()
#                             optimizer.step()

#                     # statistics
#                     running_loss += loss.item() * inputs.size(0)
#                     running_corrects += torch.sum(preds == labels.data)
#                 if phase == 'train':
#                     scheduler.step()

#                 epoch_loss = running_loss / dataset_sizes[phase]
#                 epoch_acc = running_corrects.double() / dataset_sizes[phase]

#                 print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#                 # deep copy the model
#                 if phase == 'val' and epoch_acc > best_acc:
#                     best_acc = epoch_acc
#                     print('save model')
#                     torch.save(model.state_dict(), '/home/phongnn/test/test/fast-reid/view_train_conv12.pt')

#             print()

#         time_elapsed = time.time() - since
#         print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#         print(f'Best val Acc: {best_acc:4f}')
#        # print(f'Learning rate = {}')
#         # load best model weights
#         #model.load_state_dict(torch.load(best_model_params_path))
#         return model

# state_dict = torch.load('/home/phongnn/test/test/fast-reid/tools/osnet_x1_0_imagenet.pth')
# # dict_conv_first = {}
# # dict_features = {}
# # dict_conv_last = {}

# dict_conv5 = {}
# dict_conv1 = {}
# dict_conv2 = {}
# dict_conv3 = {}
# dict_conv4 = {}
# # for key, value in state_dict.items():
# #     # if key.startswith('module.first'):
# #     #     dict_conv_first[key[18:]] = value
# #     # if key.startswith('module.features'):
# #     #     dict_features[key[16:]] = value
# #     # if key.startswith('module.conv_last'):
# #     #     dict_conv_last[key[17:]] = value

# #     if key.startswith('conv5'):
# #         dict_conv5[key[6:]] = value
# #     elif key.startswith('conv1'):
# #         dict_conv1[key[6:]] = value
# #     elif key.startswith('conv2'):
# #         dict_conv2[key[6:]] = value
# #     elif key.startswith('conv3'):
# #         dict_conv3[key[6:]] = value
# #     elif key.startswith('conv4'):
# #         dict_conv4[key[6:]] = value
# model_ft = build_osnet_backbone()
# model_ft.load_state_dict(state_dict, strict = False)
# #model_ft.load_state_dict(torch.load('/mnt/hdd3tb/Users/phongnn/test/yolo/fast-reid/freeze_conv1_conv2.pt'))
# # model_ft.conv_last1.load_state_dict(dict_conv_last)
# # model_ft.conv_last2.load_state_dict(dict_conv_last)
# # model_ft.conv_last3.load_state_dict(dict_conv_last)

# # model_ft.conv1.load_state_dict(dict_conv1, strict = True)
# # model_ft.conv2.load_state_dict(dict_conv2, strict = True)
# # model_ft.conv3.load_state_dict(dict_conv3, strict = True)
# #model_ft.first_conv.load_state_dict(dict_conv_first, strict= True)
# # model_ft.conv1.load_state_dict(dict_conv_first, strict = True)
# #model_ft.features.load_state_dict(dict_features, strict = True)
# #model_ft.conv1.load_state_dict(dict_conv_first, strict= True)

# # model_ft.conv1_clone.load_state_dict(dict_conv1)
# # model_ft.conv2_clone.load_state_dict(dict_conv2)
# #model_ft.load_state_dict(torch.load('/mnt/hdd3tb/Users/phongnn/test/yolo/fast-reid/best_model_params.pt'))
# #print(model_ft.conv51.state_dict)


# # for param in model_ft.parameters():
# #     param.requires_grad = False
# # model_ft.eval()
# # model_ft.view_predictor1 = nn.Conv2d(in_channels= 256, out_channels= 128, kernel_size= (3, 3), stride= 2, padding= 1)
# # model_ft.view_predictor2 = nn.Conv2d(in_channels= 128, out_channels= 256, kernel_size= (3, 3), stride= 2, padding= 1)
# # model_ft.view_predictor3 = nn.Conv2d(in_channels= 256, out_channels= 512,kernel_size= (3, 3), stride= 2, padding= 1)
# # model_ft.globalpool = nn.AdaptiveAvgPool2d(1)
# # model_ft.view_predictor4 = nn.Linear(in_features= 1*1*512, out_features= 3)
# for param in model_ft.conv4.parameters():
#     param.requires_grad = False
# for param in model_ft.conv5.parameters():
#     param.requires_grad = False
# # for param in model_ft.first_conv.parameters():
# #     param.requires_grad = False
# # for param in model_ft.features.parameters():
# #     param.requires_grad = False

       
# # for param in model_ft.conv_last1.parameters():
# #     param.requires_grad = False
# # for param in model_ft.conv_last2.parameters():
# #     param.requires_grad = False
# # for param in model_ft.conv_last3.parameters():
# #     param.requires_grad = False

# model_ft = model_ft.to(device)
# criterion = nn.CrossEntropyLoss()
# # BASE_LR: 0.00035
# #   WEIGHT_DECAY: 0.0005
# #   WEIGHT_DECAY_NORM: 0.0005
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0002, weight_decay = 0.0005)
# #optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.0002, weight_decay= 0.0005, momentum= 0.9)
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones = [20, 40, 60], gamma=0.1)
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                       num_epochs=50)
# # model_ft.eval()
# # img = cv2.imread('/mnt/hdd3tb/Users/phongnn/test/market1501/Market-1501-v15.09.15/gt_bbox/0048_c2s1_004401_00.jpg')
# # img = np.transpose(img, (2, 0, 1))
# # img = np.expand_dims(img, 0)
# # img = torch.from_numpy(img)
# # img = img.type(torch.cuda.FloatTensor)
# # print(model_ft(img)[1])
# # train_loader = dataloaders['train']
# # it = iter(train_loader)
# # first, _ = next(it)
# # first = first.type(torch.FloatTensor)
# # #print(model_ft(first.to('cuda:0'))[1])
# # img2, _ = image_datasets['train'][1]
# # img2 = img2.unsqueeze(0)
# # img2 = img2.type(torch.FloatTensor)
# # print(img2.shape)
# # print(img2)
# # img2 = torch.from_numpy(np.array(img2))
# # print(model_ft(img2.to('cuda:0')))
# # path = '/mnt/hdd3tb/Users/phongnn/test/market1501/Market-1501-v15.09.15/gt_bbox/0001_c1s1_001051_00.jpg'
# # file_name = '0001_c1s1_001051_00.jpg'
# # img3 = cv2.imread(path)
# # heatmaps = ['/mnt/hdd3tb/Users/phongnn/test/market1501/heatmap2' + file_name[:-4] + '_heatmap_' + str(i) + '.jpg' for i in range(1, 18)]
# # #heatmap co shape la (128, 64, 3) - (height-width-channel)
# # #chuyen ve anh xam
# # heatmap_grays = []
# # for heatmap in heatmaps:
# #     heatmap = cv2.imread(heatmap)
# #     print(heatmap)
# #     heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
# #     #heatmap_gray = heatmap_gray[:, :, np.newaxis]
# #     heatmap_grays.append(heatmap_gray)
# # heatmap_grays = np.stack(heatmap_grays, axis = -1)
# # img3 = np.concatenate((img3, heatmap_grays), axis = -1)
# # img3 = np.transpose(img3,(2, 0, 1))
# # img3 = torch.from_numpy(img3)
# # model_ft(img3.to('cuda:0'))