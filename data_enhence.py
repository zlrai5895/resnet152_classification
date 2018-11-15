# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:03:17 2018

@author: ws
"""


import pandas as pd
import numpy as np
import os
from random import randint
import random

csv_file='/home/ws/3z_gan/dataset/train/train_2000.csv'
csv=pd.read_csv(csv_file)
names=list(csv['name'])
kinds=list(csv['label'])

desert=[]
ocean=[]
mountain=[]
city=[]
farm=[]
lake=[]

ori_length=[636,463,456,218,172,55]
for i in range(len(names)):
    if kinds[i]=='DESERT':
        desert.append(names[i])
    if kinds[i]=='FARMLAND':
        farm.append(names[i])
    if kinds[i]=='CITY':
        city.append(names[i])
    if kinds[i]=='LAKE':
        lake.append(names[i])
    if kinds[i]=='MOUNTAIN':
        mountain.append(names[i])
    if kinds[i]=='OCEAN':
        ocean.append(names[i])
        
farm=farm*3
lake=lake*4
city=city*12


while(len(desert)<700):
    i=randint(0,636)
    desert.append(desert[i])

while(len(mountain)<700):
    i=randint(0,463)
    mountain.append(mountain[i])
    
while(len(ocean)<700):
    i=randint(0,456)
    ocean.append(ocean[i])


while(len(farm)<700):
    i=randint(0,218)
    farm.append(farm[i])
    
while(len(lake)<700):
    i=randint(0,172)
    lake.append(lake[i])
    
while(len(city)<700):
    i=randint(0,55)
    city.append(city[i])
    

names=desert+ocean+mountain+city+farm+lake
kinds=['DESERT']*700+['OCEAN']*700+['MOUNTAIN']*700+['CITY']*700+['FARMLAND']*700+['LAKE']*700


cc = list(zip(names, kinds))
random.shuffle(cc)
names[:], kinds[:] = zip(*cc)



dataframe = pd.DataFrame({'name':names,'label':kinds})
dataframe.to_csv("train_data.csv",index=False,sep=',')


