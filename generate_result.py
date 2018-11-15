#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:43:45 2018

@author: root
"""

import pandas as pd
import numpy as np
import os
import csv



def get_label(csv_file):
    csv=pd.read_csv(csv_file)
#    print(csv['label'].value_counts())
    label_names=csv['label'].value_counts().index
    label_ids=[x for x in range(0,len(label_names))]
    label2id=dict(zip(label_names,label_ids))
    id2label=dict(zip(label_ids,label_names))
    return label2id,id2label  

predicts=list(np.load('./pred/step_4565acc_1.0loss_tensor0.0004_params.npy'))
file_names=os.listdir('/home/ws/3z_gan/dataset/val')
label2id,id2label=get_label('/home/ws/3z_gan/dataset/train/train_2000.csv')
pre_names=[id2label[x] for x in predicts]

data=[ [file_names[i],pre_names[i]]   for i in range(len(pre_names))  ]
csv_path='./results/step_4565acc_1.0loss_tensor0.0004_params.csv'
with open(csv_path, 'w', newline='') as csvfile:
    writer  = csv.writer(csvfile)
    for row in data:
        writer.writerow(row)





#dataframe = pd.DataFrame({'name':file_names,'label':pre_names})

#dataframe.to_csv("./results/step_580acc_0.9975186104218362loss_0.0318_final_preds.csv",index=False,sep=',')



#
#filenames=sorted(os.listdir('/home/ws/3z_gan/rewrite/pred/'))
#for i in range(len(filenames)):
#    temp_predicts=list(np.load('/home/ws/3z_gan/rewrite/pred/'+filenames[i]))
#    print(np.unique(temp_predicts))
