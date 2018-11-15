#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 23:13:36 2018

@author: root
"""


import pandas as pd
import numpy as np
import os

def get_label(csv_file):
    csv=pd.read_csv(csv_file)
#    print(csv['label'].value_counts())
    label_names=csv['label'].value_counts().index
    label_ids=[x for x in range(0,len(label_names))]
    label2id=dict(zip(label_names,label_ids))
    id2label=dict(zip(label_ids,label_names))
    return label2id,id2label  


predicts=list(np.load('./pred/train_resnet152.npy'))
file_names=os.listdir('/home/ws/3z_gan/dataset/val')
label2id,id2label=get_label('/home/ws/3z_gan/dataset/train/train_2000.csv')
pre_names=[id2label[x] for x in predicts]




targets=list(np.load('./pred/train_targets_resnet152.npy'))
target_names=[id2label[x] for x in targets]


train_filenames=list(np.load('./pred/train_file_names.npy'))



different_filanames=[]
different_targets=[]

for i in range(len(train_filenames)):
    if target_names[i]!=pre_names[i]:
        different_filanames.append(train_filenames[i])
        different_targets.append(target_names[i])

np.save('./pred/resnet152_diffrent_filenames.npy',different_filanames)
np.save('./pred/resnet152_different_targets.npy',different_targets)