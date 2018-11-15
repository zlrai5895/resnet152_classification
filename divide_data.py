#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 18:54:55 2018

@author: ws
"""


import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import string
import numpy as np
from PIL import Image
import pandas as pd
import random

csv_file='/home/ws/3z_gan/dataset/train/train_2000_pre.csv'
csv=pd.read_csv(csv_file)
names=list(csv['name'])
labels=list(csv['label'])


data_names={'DESERT':[],'FARMLAND':[],'CITY':[],'LAKE':[],'MOUNTAIN':[],'OCEAN':[]}


#ori_length=[636,463,456,218,172,55]
for i in range(len(labels)):
    data_names[labels[i]].append(names[i])


train_names=[]
val_names=[]

train_labels=[]
val_labels=[]

for key in data_names.keys():
    length=len(data_names[key])
    train_length=int(length*0.8)
    val_length=length-train_length
    train_names.extend(data_names[key][:train_length])
    train_labels.extend([key]*train_length)
    
    
    val_names.extend(data_names[key][train_length:])
    val_labels.extend([key]*(val_length))


cc = list(zip(train_names, train_labels))
random.shuffle(cc)
train_names[:], train_labels[:] = zip(*cc)


dd = list(zip(val_names, val_labels))
random.shuffle(dd)
val_names[:], val_labels[:] = zip(*dd)
    
    
train_dataframe = pd.DataFrame({'name':train_names,'label':train_labels})
train_dataframe.to_csv("train_csv_file.csv",index=False,sep=',')    


train_dataframe = pd.DataFrame({'name':val_names,'label':val_labels})
train_dataframe.to_csv("val_csv_file.csv",index=False,sep=',')    


