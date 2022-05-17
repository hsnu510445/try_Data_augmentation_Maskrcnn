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
import os

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
        
        if np.sum(crop_mask)<1000:
            print(f"skip too small {np.sum(crop_mask)}")
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


