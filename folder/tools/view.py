import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import time
import json
import os, sys
import torch.nn.functional as F
from yolo_pose_onnx import *
sys.path.append('/home/phongnn/test/test/fast-reid')
from tqdm import tqdm
from PIL import Image
from tempfile import TemporaryDirectory
from fastreid.modeling.backbones import build_osnet_backbone, build_shufflenetv2_backbone
import cv2

data_transforms ={
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),
        #vertical(0.5),t
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(p = 0.5),
        transforms.RandomAffine(10),

        #transforms.ColorJitter(brightness = 0.5, hue = .3),
        #transforms.RandomPosterize(bits= 2)
        #transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
    #     transforms.RandomPosterize(bits= 2)

    #     #transforms.Resize(256),
    #     #transforms.CenterCrop(224),
    #     #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def make_dataset(directory):
    class_to_idx = {'0_front':0,
                    '1_side':1,
                    '2_back':2}
    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks= True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                instances.append(item)

                if target_class not in available_classes:
                    available_classes.add(target_class)
    
    return instances

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.classes = ['0_front', '1_side', '2_back']
        self.class_to_idx = {'0_front':0,
                            '1_side':1,
                            '2_back':2}
        self.root = root
        self.transform = transform
        self.samples = make_dataset(root)
    def __getitem__(self, index):
        path, target = self.samples[index]
        file_name = path.split('/')[-1]
        img = cv2.imread(path)
        img = np.transpose(img,(2, 0, 1))
        # #img = self.transform(img)
        img = torch.from_numpy(img)
        img = self.transform(img)
        target = torch.tensor(target)
        #target = self.transform(target)
        return img, target
    def __len__(self):
        return len(self.samples)

data_dir = '/home/phongnn/test/test/Market1203/Market1203_orient_3classes_split'    
image_datasets = {x: CustomDataset(os.path.join(data_dir, x), transform= data_transforms[x])
                  for x in ['train', 'val']} 
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=2)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device).float()
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    print('save model')
                    torch.save(model.state_dict(), 'view_resnet.pt')

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
       # print(f'Learning rate = {}')
        # load best model weights
        #model.load_state_dict(torch.load(best_model_params_path))
        return model

# Initialize model
# weights = ResNet50_Weights.DEFAULT
#model = resnet50(weights = weights)
model = resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
# model = resnet50()
model = model.to(device)
# criterion = nn.CrossEntropyLoss()

# optimizer_ft = optim.Adam(model.parameters(), lr=0.0002, weight_decay = 0.0005)
# exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones = [20, 40, 60], gamma=0.1)
# model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
#                      num_epochs=50)


def predict(model, img):
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    img = img.type(torch.cuda.FloatTensor)
    pred = model(img).softmax(1)
    class_id = pred.argmax().item()
    return class_id

model.load_state_dict(torch.load('/home/phongnn/test/test/fast-reid/view_resnet.pt'))
model.eval()
img = cv2.imread('/home/phongnn/test/test/market1501/Market-1501-v15.09.15/bounding_box_train/0002_c1s1_000551_01.jpg')
predict(model, img)

root = '/home/phongnn/test/test/market1501/Market-1501-v15.09.15'

with open('/home/phongnn/test/test/market1501/Market-1501-v15.09.15/label1.json', 'w') as f:
    #f.write('[')
    dictt = {}
    for folder in sorted(os.listdir(root)):
      if folder not in ['gt_query', 'label.json', 'readme.txt']:
        print(folder)
        folder_path = os.path.join(root, folder)
        for img in sorted(os.listdir(folder_path)):
          if img[0] != 'T':
            img_path = os.path.join(folder_path, img)
            image = cv2.imread(img_path)
            try:
                label = predict(model, image)
            except:
                print(img_path)
            dictt[img_path] = label
            # data = {
            #     "img_path": img_path,
            #     "label": label
            # }
            # json.dump(data, f)
            # f.write(',')
            # f.write("\n")
          else:
            continue
    f.dump(dictt, f)
# with open('/home/phongnn/test/test/market1501/Market-1501-v15.09.15/label.json', 'a') as f:
#     f.write("\n]")
