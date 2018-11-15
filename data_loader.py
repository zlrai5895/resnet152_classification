# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:31:38 2018

@author: ws
"""
########################此代码用于加载数据####################################

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import string
import numpy as np
from PIL import Image
import pandas as pd


#将标签和索引一一对应
def get_label(csv_file):
    csv=pd.read_csv(csv_file)
#    print(csv['label'].value_counts())
    label_names=csv['label'].value_counts().index
    label_ids=[x for x in range(0,len(label_names))]
    label2id=dict(zip(label_names,label_ids))
    id2label=dict(zip(label_ids,label_names))
    return label2id,id2label    
    #label2id:{'MOUNTAIN': 1, 'FARMLAND': 3, 'CITY': 5, 'LAKE': 4, 'OCEAN': 2, 'DESERT': 0}  
    #id2label:{1: 'DESERT', 2: 'MOUNTAIN', 3: 'OCEAN', 4: 'FARMLAND', 5: 'LAKE', 6: 'CITY'}
    


###########建立数据集  继承自torch.utils.data.Dataset#########################
#__init__   __getitem__ __len__
class trainDataset(data.Dataset):
    def __init__(self,  root_dir,csv_file, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.csv=csv_file

    def __getitem__(self, index):   #返回第index条数据
        img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[index, 0])
        image = Image.open(img_name)
        label_name = self.landmarks_frame.iloc[index, 1]
        label2id,id2label=get_label(self.csv)
        label=label2id[label_name]

        if self.transform:
            image = self.transform(image)
            
        return image, label,img_name
    
    def __len__(self):
        return  len(self.landmarks_frame)



class valDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.filenames=os.listdir(root_dir)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,self.filenames[index])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)
            
        return image
    
    def __len__(self):
        return  len(self.filenames)


#########################建立数据集加载器####################################
def get_train_loader(csv_file,root,transform, batch_size, shuffle, num_workers):
    data_set = trainDataset(root,csv_file,transform=transform)
        
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader

def get_val_loader(csv_file,root,transform, batch_size, shuffle, num_workers):
    data_set = trainDataset(root,csv_file,transform=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,)
    return data_loader

def get_test_loader(root,transform, batch_size, shuffle, num_workers):
    data_set = valDataset(root,transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,)
    return data_loader