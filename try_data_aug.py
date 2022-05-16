# %%
# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split 
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm.auto import tqdm

import cv2
import matplotlib.pyplot as plt

import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

# %%
ROOT_DIR='dataset_maskrcnn'
DATA_AUG_DIR=f"{ROOT_DIR}/data_augmentation/"

# %%
Image.open(f'{ROOT_DIR}/PNGImages/02.png')

# %%
mask = Image.open(f'{ROOT_DIR}/LarvaMasks/label_02_mask.png')
# each mask instance has a different color, from zero to N, where
# N is the number of instances. In order to make visualization easier,
# let's adda color palette to the mask.
mask
# mask.mode
# 要從L換成P，不然好像是會報錯
mask.convert('P')
mask.putpalette([
    0, 0, 0, # black background
    128, 0, 0, # index 1 is red
    0, 128, 0, # index 2 is green
    128, 128, 0, # index 3 is yellow
])

mask
# %%
# 因為SSD的數據不能直接用Pytorch的數據增強，所以先存一波已經變換過的圖
# 先改寫一些transform 函式 -> 改寫失敗，因為型別class錯誤會導致錯誤
# 好像PIL是四維的(RGBA)，最後一維都是225，換成mask
# test_img=Image.open(f'{ROOT_DIR}/PNGImages/02.png')
# test_mask = Image.open(f'{ROOT_DIR}/LarvaMasks/label_02_mask.png')
# np.array(test_img).shape->(2076, 3088, 4)
# np.array(test_mask).shape->(2076, 3088)
# test_img_arr=np.array(test_img)
# test_img_arr[:,:,3]=test_img_arr[:,:,3]-np.array(test_mask)
# test_img_new = Image.fromarray(test_img_arr) 
# 把mask偷渡過去
def encode(img,mask):
    '''
    input :
    PIL 圖片(height,width,4)
    mask(height,width)
    
    output : 
    PIL 圖片(height,width,4)
    '''
    img_arr=np.array(img)
    img_arr[:,:,3]=img_arr[:,:,3]-np.array(mask)#較可視:255去減mask
    img_new = Image.fromarray(img_arr) 
    return img_new

def decode(img):
    '''
    input :
    PIL 圖片(height,width,4)
    
    output : 
    PIL 圖片(height,width,4)
    mask(height,width)
    '''
    img_arr=np.array(img)
    mask=np.ones((img_arr[:,:,3].shape[0],img_arr[:,:,3].shape[1]),dtype="uint8")*255-img_arr[:,:,3]#較可視
    img_arr[:,:,3]=np.ones((img_arr[:,:,3].shape[0],img_arr[:,:,3].shape[1]),dtype="uint8")*255
    img_new = Image.fromarray(img_arr) 
    mask_new = Image.fromarray(mask) 
    return img_new,mask_new

def vis(mask):
    mask.convert('P')
    mask.putpalette([
        0, 0, 0, # black background
        128, 0, 0, # index 1 is red
        0, 128, 0, # index 2 is green
        128, 128, 0, # index 3 is yellow
    ])
    return mask

# %%
# 來輸出一波圖片
# 是否輸出可式化label
visualize=True
origin_img_list = list(sorted(os.listdir(os.path.join(ROOT_DIR, 'PNGImages'))))
origin_mask_list = list(sorted(os.listdir(os.path.join(ROOT_DIR, 'LarvaMasks'))))
shape_size=512
for origin_img_num in range(origin_img_list.__len__()):
    count=0
    origin_IMG=Image.open(f'{ROOT_DIR}/PNGImages/{origin_img_list[origin_img_num]}')
    origin_mask=Image.open(f'{ROOT_DIR}/LarvaMasks/{origin_mask_list[origin_img_num]}')

    while True:
        if count>1000:
            # 一張圖片變50張就好
            break
        
        subIMG=encode(origin_IMG,origin_mask)
        # 
        train_tfm = transforms.Compose([
            transforms.RandomApply([
                    transforms.RandomPerspective(p=1.0,fill=(0, 0, 0, 255)), 
                    transforms.RandomAffine(degrees=0, shear= 25,fill=(0, 0, 0, 255)),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomVerticalFlip(p=1.0),
                    # transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 5.0)), #加雜訊會有問題
                    transforms.RandomRotation(degrees=30,fill=(0, 0, 0, 255)),
                ],p=0.8),
            
            transforms.RandomCrop((shape_size, shape_size)),
        ])
        subIMG=train_tfm(subIMG)
        crop_img,crop_mask=decode(subIMG)
        # crop_mask[crop_mask >10] = 0
        if np.sum(crop_mask)==0:
            print("skip none")
            continue
        
        # 檢查標籤    
        crop_mask_arr=np.array(crop_mask) 
        xmax=0
        xmin=shape_size
        ymax=0
        ymin=shape_size
        for x in range(shape_size):
            for y in range(shape_size):
                if crop_mask_arr[x][y]!=0:
                    if x>xmax:
                        xmax=x
                    if x<xmin:
                        xmin=x
                    if y>ymax:
                        ymax=y
                    if y<ymin:
                        ymin=y
        if xmin==xmax:
            print("skip same x")
            continue
        if ymin==ymax:
            print("skip same y")
            continue
        
        # 存圖片
        if not os.path.isdir(f"{DATA_AUG_DIR}"):
            os.makedirs(f"{DATA_AUG_DIR}")
        if not os.path.isdir(f"{DATA_AUG_DIR}/LarvaMasks"):
            os.makedirs(f"{DATA_AUG_DIR}/LarvaMasks")
        if not os.path.isdir(f"{DATA_AUG_DIR}/PNGImages"):
            os.makedirs(f"{DATA_AUG_DIR}/PNGImages")
        
        crop_img.save(f"{DATA_AUG_DIR}/PNGImages/{origin_img_list[origin_img_num][:-4]}_{str(count).zfill(4)}.png","png")
        crop_mask.save(f"{DATA_AUG_DIR}/LarvaMasks/{origin_mask_list[origin_img_num][:-4]}_{str(count).zfill(4)}.png","png")
        if visualize:
            if not os.path.isdir(f"{DATA_AUG_DIR}/LarvaMasks_vis"):
                os.makedirs(f"{DATA_AUG_DIR}/LarvaMasks_vis")
            crop_mask_vis=vis(crop_mask)
            crop_mask_vis.save(f"{DATA_AUG_DIR}/LarvaMasks_vis/{origin_img_list[origin_img_num][:-4]}_{str(count).zfill(4)}_vis.png","png")

        count=count+1
# http://noahsnail.com/2020/06/12/2020-06-12-%E7%8E%A9%E8%BD%ACpytorch%E4%B8%AD%E7%9A%84torchvision.transforms/
# %%
# 用來顯示所有數字
# import sys
# np.set_printoptions(threshold=sys.maxsize)
# print(test_img_arr[:,:,3])
# %%
import os
import numpy as np
import torch
from PIL import Image


class LarvaDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'LarvaMasks'))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, 'PNGImages', self.imgs[idx])
        mask_path = os.path.join(self.root, 'LarvaMasks', self.masks[idx])
        img = Image.open(img_path).convert('RGB')
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img,mask, target

    def __len__(self):
        return len(self.imgs)

# %%
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    return model

# %%
# %shell
import subprocess
# Download TorchVision repo to use some files from
# references/detection
subprocess.call("git clone https://github.com/pytorch/vision.git", shell=True)
subprocess.call("cd vision", shell=True)
subprocess.call("git checkout v0.8.2", shell=True)

subprocess.call("cp references/detection/utils.py ../", shell=True)
subprocess.call("cp references/detection/transforms.py ../", shell=True)
subprocess.call("cp references/detection/coco_eval.py ../", shell=True)
subprocess.call("cp references/detection/engine.py ../", shell=True)
subprocess.call("cp references/detection/coco_utils.py ../", shell=True)



# %%
from vision.references.detection.engine import train_one_epoch, evaluate
import utils
import vision.references.detection.transforms as T


def get_transform(train):
    transforms = []
    # TODO data augmentation
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# %%
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = LarvaDataset(f'{ROOT_DIR}', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn
)
# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions

# %%
# use our dataset and defined transformations
dataset = LarvaDataset(f'{ROOT_DIR}', get_transform(train=True))
dataset_test = LarvaDataset(f'{ROOT_DIR}', get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
# %%
# 分隔線-------------------------------------------------------------------------------
# %%
# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.RandomCrop((128, 128)),
    transforms.RandomPerspective(p=1.0), 
    transforms.RandomAffine(degrees=0, shear= 25),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 5.0)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.RandomCrop((128, 128)),
    transforms.ToTensor(),
])

# %%
# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 128

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = LarvaDataset(f'{ROOT_DIR}' ,transforms=train_tfm)
# Random split
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size])


# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# %%
print('Original image: ')
img = cv2.imread(f'{ROOT_DIR}/PNGImages/02.png')[:,:,::-1]
plt.imshow(img)
plt.show()

print("Augmented image: ")
for i in range(7):  # display applied augmentations (skip all rotations)
    img = cv2.imread(f'{ROOT_DIR}/PNGImages/02.png')[:,:,::-1]
    print(train_tfm[i])
    plt.imshow(img)
    plt.show()
# %%
